#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits>  // For INFINITY

#define N 8192
#define M 16
#define T 1024

static void load_data(const char* filename, double data[N][M]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening training file");
        return;
    }

    char line[1024];
    char* token;
    int row = 0;

    while (fgets(line, sizeof(line), file) && row < N) {
        token = strtok(line, ",");
        for (int col = 0; token && col < M; col++) {
            data[row][col] = atof(token);
            token = strtok(NULL, ",");
        }
        row++;
    }
    fclose(file);
}

// Function to evaluate accuracy of predictions
static double evaluate_accuracy(int* pred_labels, int* true_labels) {
    int correct = 0;
    for (int i = 0; i < T; i++) {
        if (pred_labels[i] == true_labels[i]) {
            correct++;
        }
    }
    return (double)correct / T;
}

// Function to find K nearest neighbors of a test instance
static void findKNN(double test_instance[], double train_chunk[][M], int* Knn, int P) {
    double min_dist = std::numeric_limits<double>::infinity();
    double Class_label;

    for (int j = 0; j < N / P; j++) {
        double dist = 0.0;
        for (int k = 0; k < M - 1; k++) {
            dist += (test_instance[k] - train_chunk[j][k]) * (test_instance[k] - train_chunk[j][k]);
        }
        if (dist < min_dist) {
            min_dist = dist;
            Class_label = train_chunk[j][M - 1];
        }
    }
    *Knn = (int)Class_label;
}

int main(int argc, char** argv) {
    int rank, P;
    int mpi_status;
    double start_time, end_time;
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    double (*train_data)[M] = (double (*)[M])malloc(N * M * sizeof(double));
    double (*train_chunk)[M] = (double (*)[M])malloc((N / P) * M * sizeof(double));
    double (*test_data)[M] = (double (*)[M])malloc(T * M * sizeof(double));

    if (train_data == NULL || train_chunk == NULL || test_data == NULL) {
        printf("Memory allocation failed\n");
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        start_time = MPI_Wtime();  // Start timing
        load_data("train.csv", train_data);
        load_data("test.csv", test_data);
    }

    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast test data to all processes
    mpi_status = MPI_Bcast(test_data, T * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute training data chunks to processes
    mpi_status = MPI_Scatter(train_data, (N / P) * M, MPI_DOUBLE, train_chunk, (N / P) * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process finds KNN for its chunk
    int knn[T];
    for (int i = 0; i < T; i++) {
        findKNN(test_data[i], train_chunk, &knn[i], P);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int* recv_knn = (int*)malloc(T * P * sizeof(int));

    mpi_status = MPI_Gather(knn, T, MPI_INT, recv_knn, T, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_status != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: MPI_Gather failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, mpi_status);
        return mpi_status;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // Determine predicted labels using majority voting
        int pred_labels[T];
        int test_labels[T];

        for (int i = 0; i < T; i++) {
            int class_counts[2] = { 0 };
            for (int j = 0; j < P; j++) {
                class_counts[recv_knn[i * P + j]]++;
            }
            pred_labels[i] = (class_counts[0] > class_counts[1]) ? 0 : 1;
            test_labels[i] = (int)test_data[i][M - 1];
        }

        // Calculate and print accuracy
        double accuracy = evaluate_accuracy(pred_labels, test_labels);
        printf("Accuracy: %f\n", accuracy);

        end_time = MPI_Wtime();  // End timing
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    // Free dynamically allocated memory
    free(train_data);
    free(train_chunk);
    free(test_data);
    free(recv_knn);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
