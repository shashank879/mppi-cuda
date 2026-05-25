// csrc/mppi_rollout.cu — v0: fused MPPI rollout kernel for 7-DoF Franka reach.
//
// Design (see docs/kernel_design.md):
//   - 1 thread per rollout. State (q, qdot) and FK transforms live in
//     registers; only the unique noise stripe and the final cost touch
//     global memory.
//   - Dynamics + cost are inlined; no template parameters yet (v0).
//   - Bit-exact to the PyTorch baseline within fp32 tolerance.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

// Franka modified-DH parameters: (a, d, alpha) per joint.
// MUST match mppi_cuda/kinematics.py::FRANKA_DH exactly.
__constant__ float FRANKA_DH[7][3] = {
    { 0.0f,     0.333f,  0.0f                  },
    { 0.0f,     0.0f,   -1.5707963267948966f   },
    { 0.0f,     0.316f,  1.5707963267948966f   },
    { 0.0825f,  0.0f,    1.5707963267948966f   },
    { -0.0825f, 0.384f, -1.5707963267948966f   },
    { 0.0f,     0.0f,    1.5707963267948966f   },
    { 0.088f,   0.0f,    1.5707963267948966f   },
};

#define FRANKA_FLANGE_D 0.107f
#define N_JOINTS 7

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// T_out = T_in * dh(a, d, alpha, theta). Row-major 4x4.
// Used in alternating-buffer style so each call is `apply_dh(B, A, ...)`
// and the caller swaps the role of A and B for the next joint.
__device__ __forceinline__ void apply_dh(
    float T_out[16], const float T_in[16],
    float a, float d, float alpha, float theta)
{
    const float ca = cosf(alpha), sa = sinf(alpha);
    const float ct = cosf(theta), st = sinf(theta);

    // DH matrix (modified-DH, Craig convention):
    //  [  ct        -st         0     a    ]
    //  [  st*ca     ct*ca      -sa   -d*sa ]
    //  [  st*sa     ct*sa       ca    d*ca ]
    //  [  0         0           0     1    ]
    //
    // T_out[i, j] = sum_k T_in[i, k] * Ti[k, j]. Row-by-row to keep
    // 4 floats in flight per iteration.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const float r0 = T_in[i*4 + 0];
        const float r1 = T_in[i*4 + 1];
        const float r2 = T_in[i*4 + 2];
        const float r3 = T_in[i*4 + 3];

        T_out[i*4 + 0] = r0 * ct      + r1 * (st*ca) + r2 * (st*sa);
        T_out[i*4 + 1] = r0 * (-st)   + r1 * (ct*ca) + r2 * (ct*sa);
        T_out[i*4 + 2] =                 r1 * (-sa)  + r2 * ca;
        T_out[i*4 + 3] = r0 * a       + r1 * (-d*sa) + r2 * (d*ca)  + r3;
    }
}

// EE position from joint angles (drops the rotation; cost only needs position).
__device__ __forceinline__ void forward_kinematics_pos(
    const float q[N_JOINTS], float ee_pos[3])
{
    float T_a[16];
    float T_b[16];

    // T_a = identity
    #pragma unroll
    for (int i = 0; i < 16; i++) T_a[i] = 0.0f;
    T_a[0] = T_a[5] = T_a[10] = T_a[15] = 1.0f;

    // 7 DH transforms, alternating buffers to avoid copies.
    apply_dh(T_b, T_a, FRANKA_DH[0][0], FRANKA_DH[0][1], FRANKA_DH[0][2], q[0]);
    apply_dh(T_a, T_b, FRANKA_DH[1][0], FRANKA_DH[1][1], FRANKA_DH[1][2], q[1]);
    apply_dh(T_b, T_a, FRANKA_DH[2][0], FRANKA_DH[2][1], FRANKA_DH[2][2], q[2]);
    apply_dh(T_a, T_b, FRANKA_DH[3][0], FRANKA_DH[3][1], FRANKA_DH[3][2], q[3]);
    apply_dh(T_b, T_a, FRANKA_DH[4][0], FRANKA_DH[4][1], FRANKA_DH[4][2], q[4]);
    apply_dh(T_a, T_b, FRANKA_DH[5][0], FRANKA_DH[5][1], FRANKA_DH[5][2], q[5]);
    apply_dh(T_b, T_a, FRANKA_DH[6][0], FRANKA_DH[6][1], FRANKA_DH[6][2], q[6]);

    // T_b is the final transform after joint 7. Apply flange offset:
    // ee_pos = T_b * [0, 0, FRANKA_FLANGE_D, 1]^T
    //        = FRANKA_FLANGE_D * T_b[:, 2] + T_b[:, 3]
    ee_pos[0] = FRANKA_FLANGE_D * T_b[0*4 + 2] + T_b[0*4 + 3];
    ee_pos[1] = FRANKA_FLANGE_D * T_b[1*4 + 2] + T_b[1*4 + 3];
    ee_pos[2] = FRANKA_FLANGE_D * T_b[2*4 + 2] + T_b[2*4 + 3];
}

// Per-thread: roll out one trajectory for H steps, accumulate cost, write out.
__global__ void mppi_rollout_kernel(
    const float* __restrict__ x0,         // (14,)
    const float* __restrict__ U_nominal,  // (H, 7)
    const float* __restrict__ noise,      // (K, H, 7)
    const float* __restrict__ target_pos, // (3,)
    const float* __restrict__ q_min,      // (7,)
    const float* __restrict__ q_max,      // (7,)
    const float* __restrict__ qdot_max,   // (7,)
    float* __restrict__ costs_out,        // (K,)
    int K, int H,
    float u_min, float u_max,
    float dt,
    float w_pos, float w_u, float w_qdot, float w_lim,
    float terminal_scale)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Per-thread state in registers.
    float q[N_JOINTS], qdot[N_JOINTS];
    #pragma unroll
    for (int j = 0; j < N_JOINTS; j++) {
        q[j]    = x0[j];
        qdot[j] = x0[N_JOINTS + j];
    }

    // Stage constants in registers. All threads read same indices →
    // broadcast in L1/L2 trivially, no shared-memory cooperation needed.
    float tgt[3]   = { target_pos[0], target_pos[1], target_pos[2] };
    float qmin[N_JOINTS], qmax[N_JOINTS], qdmax[N_JOINTS];
    #pragma unroll
    for (int j = 0; j < N_JOINTS; j++) {
        qmin[j]  = q_min[j];
        qmax[j]  = q_max[j];
        qdmax[j] = qdot_max[j];
    }

    float cost = 0.0f;

    // Sequential time loop. Each iteration:
    //   compute u_t (clamp(U_nom + noise))
    //   accumulate running_cost(x, u_t)
    //   advance dynamics
    for (int t = 0; t < H; t++) {
        // Effective control = clamped (U_nominal[t] + noise[k, t]).
        float u[N_JOINTS];
        #pragma unroll
        for (int j = 0; j < N_JOINTS; j++) {
            const float raw =
                U_nominal[t * N_JOINTS + j] +
                noise[(k * H + t) * N_JOINTS + j];
            u[j] = clampf(raw, u_min, u_max);
        }

        // Running cost components.
        // (a) position via FK
        float ee[3];
        forward_kinematics_pos(q, ee);
        const float dx = ee[0] - tgt[0];
        const float dy = ee[1] - tgt[1];
        const float dz = ee[2] - tgt[2];
        const float pos_err_sq = dx*dx + dy*dy + dz*dz;

        // (b) control magnitude, (c) joint velocity magnitude
        float u_sq = 0.0f, qdot_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < N_JOINTS; j++) {
            u_sq    += u[j]    * u[j];
            qdot_sq += qdot[j] * qdot[j];
        }

        // (d) joint-limit barrier
        float lim_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < N_JOINTS; j++) {
            const float below = fmaxf(qmin[j] - q[j], 0.0f);
            const float above = fmaxf(q[j] - qmax[j], 0.0f);
            lim_sq += below*below + above*above;
        }

        cost += w_pos * pos_err_sq
              + w_u    * u_sq
              + w_qdot * qdot_sq
              + w_lim  * lim_sq;

        // Dynamics: semi-implicit Euler with velocity clamp.
        #pragma unroll
        for (int j = 0; j < N_JOINTS; j++) {
            float qd_new = qdot[j] + dt * u[j];
            qd_new       = clampf(qd_new, -qdmax[j], qdmax[j]);
            qdot[j]      = qd_new;
            q[j]         = q[j] + dt * qd_new;
        }
    }

    // Terminal cost: weighted position error at post-final state.
    float ee[3];
    forward_kinematics_pos(q, ee);
    const float dx = ee[0] - tgt[0];
    const float dy = ee[1] - tgt[1];
    const float dz = ee[2] - tgt[2];
    cost += terminal_scale * w_pos * (dx*dx + dy*dy + dz*dz);

    costs_out[k] = cost;
}

// ------- Python binding -------

torch::Tensor mppi_rollout(
    torch::Tensor x0,
    torch::Tensor U_nominal,
    torch::Tensor noise,
    torch::Tensor target_pos,
    torch::Tensor q_min,
    torch::Tensor q_max,
    torch::Tensor qdot_max,
    double u_min, double u_max,
    double dt,
    double w_pos, double w_u, double w_qdot, double w_lim,
    double terminal_scale)
{
    TORCH_CHECK(x0.is_cuda(),         "x0 must be CUDA");
    TORCH_CHECK(U_nominal.is_cuda(),  "U_nominal must be CUDA");
    TORCH_CHECK(noise.is_cuda(),      "noise must be CUDA");
    TORCH_CHECK(target_pos.is_cuda(), "target_pos must be CUDA");
    TORCH_CHECK(q_min.is_cuda() && q_max.is_cuda() && qdot_max.is_cuda(),
                "joint limits must be CUDA");

    TORCH_CHECK(x0.scalar_type() == torch::kFloat32, "x0 must be fp32");
    TORCH_CHECK(noise.scalar_type() == torch::kFloat32, "noise must be fp32");

    TORCH_CHECK(x0.is_contiguous(),        "x0 must be contiguous");
    TORCH_CHECK(U_nominal.is_contiguous(), "U_nominal must be contiguous");
    TORCH_CHECK(noise.is_contiguous(),     "noise must be contiguous");

    TORCH_CHECK(x0.dim() == 1 && x0.size(0) == 14,
                "x0 must be shape (14,), got ", x0.sizes());
    TORCH_CHECK(noise.dim() == 3 && noise.size(2) == 7,
                "noise must be shape (K, H, 7), got ", noise.sizes());
    TORCH_CHECK(U_nominal.dim() == 2 && U_nominal.size(1) == 7
                && U_nominal.size(0) == noise.size(1),
                "U_nominal must be shape (H, 7) and match noise's H, got ",
                U_nominal.sizes(), " vs noise ", noise.sizes());
    TORCH_CHECK(target_pos.numel() == 3, "target_pos must have 3 elements");
    TORCH_CHECK(q_min.numel() == 7 && q_max.numel() == 7 && qdot_max.numel() == 7,
                "joint limits must each have 7 elements");

    const int K = noise.size(0);
    const int H = noise.size(1);

    const c10::cuda::OptionalCUDAGuard guard(x0.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto costs = torch::empty({K}, x0.options());

    const int threads = 256;
    const int blocks  = (K + threads - 1) / threads;

    mppi_rollout_kernel<<<blocks, threads, 0, stream.stream()>>>(
        x0.data_ptr<float>(),
        U_nominal.data_ptr<float>(),
        noise.data_ptr<float>(),
        target_pos.data_ptr<float>(),
        q_min.data_ptr<float>(),
        q_max.data_ptr<float>(),
        qdot_max.data_ptr<float>(),
        costs.data_ptr<float>(),
        K, H,
        static_cast<float>(u_min), static_cast<float>(u_max),
        static_cast<float>(dt),
        static_cast<float>(w_pos),
        static_cast<float>(w_u),
        static_cast<float>(w_qdot),
        static_cast<float>(w_lim),
        static_cast<float>(terminal_scale)
    );
    // Optional: check for launch errors in debug builds.
    AT_CUDA_CHECK(cudaGetLastError());

    return costs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mppi_rollout", &mppi_rollout,
          "MPPI rollout (CUDA v0): per-rollout cost accumulation");
}
