import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 32, 64, 128, 256, 512, 1024]

for model in ["mixtral"]:
    if model == "mixtral":
        # NVFP4 Pingpong results (10 groups, N=4096, K=14336)
        nvfp4_times = [
            0.219944,  # batch = 1
            0.228924,  # batch = 32
            0.226622,  # batch = 64
            0.231853,  # batch = 128
            0.284941,  # batch = 256
            0.478252,  # batch = 512
            0.875735,  # batch = 1024
        ]
        # MXFP8 Cooperative results (slightly better than Pingpong for this case)
        mxfp8_times = [
            0.264242,  # batch = 1
            0.275848,  # batch = 32
            0.250447,  # batch = 64
            0.252280,  # batch = 128
            0.448731,  # batch = 256
            0.881986,  # batch = 512
            1.726160,  # batch = 1024
        ]

    # Create figure with compact size
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot lines with evenly spaced x-positions
    x_positions = np.arange(len(batch_sizes))
    ax.plot(x_positions, mxfp8_times, 'o-', color='green', linewidth=1.5, 
            markersize=4, label='MXFP8')
    ax.plot(x_positions, nvfp4_times, 's-', color='red', linewidth=1.5, 
            markersize=4, label='NVFP4')

    # Labels and formatting
    ax.set_xlabel('Batch Size', fontsize=9)
    ax.set_ylabel('Average Runtime (ms)', fontsize=9)
    ax.set_title('Grouped GEMM Performance (N=4096, K=14336, Groups=10)', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8, frameon=True, loc='upper left')

    # Set x-ticks to actual batch sizes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Tight layout
    plt.tight_layout(pad=0.3)

    # Save figure
    plt.savefig('precisions_nvfp4_mxfp8_mixtral_moe.png', dpi=300, bbox_inches='tight')
    plt.show()