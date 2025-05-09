import pandas as pd

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load the CSV file
    csv_file = "position_analysis-jointpanda_joint1-20250410-094734.csv"  # Replace with the actual path to your CSV file
    data = pd.read_csv(csv_file)

    # Plot points from the first column
    plt.plot(data.iloc[:, 0], marker='o', linestyle='-', label='First Column')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position')
    # plt.title('Plot of First Column')
    plt.legend()
    plt.grid(True)
        # Save the plot to a PNG file
    output_file = "plot_results.png"  # Replace with your desired output file name
    plt.savefig(output_file, format='png', dpi=300)
    plt.show()
