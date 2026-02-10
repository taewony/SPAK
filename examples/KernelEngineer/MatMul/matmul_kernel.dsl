system MatrixOptimizer {
    
    kernel MatMul {
      # Procedure: standard_matmul_kernel
      # Input: A[M][K], B[K][N]
      # Output: C[M][N]
      # Logic: 2D Grid, Static Shared Memory Tiling

      procedure standard_matmul_kernel:
          # 2D Grid: Each block computes C[i_tile, j_tile]
          i_tile = blockIdx.x
          j_tile = blockIdx.y
          
          acc = 0.0
          
          # K-Loop (Tiled)
          for k_tile in 0..num_k_tiles-1:
              # Load tiles to SRAM
              A_sram = load(A[i_tile, k_tile])
              B_sram = load(B[k_tile, j_tile])
              
              # Compute (MMA)
              acc += A_sram @ B_sram
          
          # Write back
          store(C[i_tile, j_tile], acc)
    }

    kernel BatchMatMul {
      # Procedure: batch_matmul_kernel  
      # Input: A[B][M][K], B[B][K][N]
      # Output: C[B][M][N]
      # Logic: 3D Grid (Batch in Z), Batch Stride Handling

      procedure batch_matmul_kernel:
          # 3D Grid: [M_Tile, N_Tile, Batch]
          i_tile = blockIdx.x
          j_tile = blockIdx.y
          b_idx  = blockIdx.z
          
          acc = 0.0
          
          # Compute pointers with Batch Stride
          # A_ptr = A_base + b_idx * stride_A
          
          for k_tile in 0..num_k_tiles-1:
              # Load from specific Batch
              A_sram = load(A[b_idx, i_tile, k_tile])
              B_sram = load(B[b_idx, k_tile, j_tile])
              
              acc += A_sram @ B_sram
          
          store(C[b_idx, i_tile, j_tile], acc)
    }
}
