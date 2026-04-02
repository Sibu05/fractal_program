#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
using namespace std;

#define DIM 768

struct cuComplex
{
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.7269, 0.1889);
    cuComplex a(jx, jy);

    for (int i = 0; i < 300; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

/* Students should parallelize this */
void kernel_omp_for(unsigned char *ptr)
{
#pragma omp parallel for

    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;                // G
            ptr[offset * 3 + 2] = 0;                // B
        }
    }
}

void kernel_omp_1d_row(unsigned char *ptr)
{
#pragma omp parallel
    {
        int nthread = omp_get_num_threads();
        int tthread = omp_get_thread_num();
        for (int y = tthread; y < DIM; y += nthread)
        {
            for (int x = 0; x < DIM; x++)
            {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset * 3 + 0] = 255 * juliaValue;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            }
        }
    }
}

void kernel_omp_1d_col(unsigned char *ptr)
{
#pragma omp parallel
    {
        int nthread = omp_get_num_threads();
        int tthread = omp_get_thread_num();
        for (int x = tthread; x < DIM; x += nthread)
        {
            for (int y = 0; y < DIM; y++)
            {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset * 3 + 0] = 255 * juliaValue;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            }
        }
    }
}

void kernel_omp_2d_row_block(unsigned char *ptr)
{
#pragma omp parallel
    {
        int nthread = omp_get_num_threads();
        int tthread = omp_get_thread_num();
        int rows_per_thread = DIM / nthread;
        int start_row = tthread * rows_per_thread;
        int end_row = (tthread == nthread - 1) ? DIM : start_row + rows_per_thread;

        for (int y = start_row; y < end_row; y++)
        {
            for (int x = 0; x < DIM; x++)
            {

                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset * 3 + 0] = 255 * juliaValue;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            }
        }
    }
}

void kernel_omp_2d_col_block(unsigned char *ptr)
{
#pragma omp parallel
    {
        int nthread = omp_get_num_threads();
        int tthread = omp_get_thread_num();
        int rows_per_thread = DIM / nthread;
        int start_row = tthread * rows_per_thread;
        int end_row = (tthread == nthread - 1) ? DIM : start_row + rows_per_thread;

        for (int x = start_row; x < end_row; x++)
        {
            for (int y = 0; y < DIM; y++)
            {

                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset * 3 + 0] = 255 * juliaValue;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            }
        }
    }
}

void kernel_serial(unsigned char *ptr)
{
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue;
            ptr[offset * 3 + 1] = 0;
            ptr[offset * 3 + 2] = 0;
        }
    }
}

/* Save image as PPM */
void save_ppm(const char *filename, unsigned char *data, int width, int height)
{
    ofstream file(filename, ios::binary);
    file << "P6\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char *>(data), width * height * 3);
    file.close();
}

bool verify_output(unsigned char *ref, unsigned char *test, const char *name)
{
    int total = DIM * DIM * 3;
    for (int i = 0; i < total; i++)
    {
        if (ref[i] != test[i])
        {
            cout << "[FAIL] " << name << " differs from serial at byte " << i << endl;
            return false;
        }
    }
    cout << "[PASS] " << name << " matches serial output." << endl;
    return true;
}

int main(void)
{
    system("mkdir -p output");

    unsigned char *image_s = new unsigned char[DIM * DIM * 3];
    unsigned char *image_1 = new unsigned char[DIM * DIM * 3];
    unsigned char *image_2 = new unsigned char[DIM * DIM * 3];
    unsigned char *image_3 = new unsigned char[DIM * DIM * 3];
    unsigned char *image_4 = new unsigned char[DIM * DIM * 3];
    unsigned char *image_5 = new unsigned char[DIM * DIM * 3];

    double start, t_serial, t1, t2, t3, t4, t5;
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_configs = 9;

    /* Serial run */
    start = omp_get_wtime();
    kernel_serial(image_s);
    t_serial = omp_get_wtime() - start;
    cout << "Serial time: " << t_serial << endl;

    /* Correctness verification */
    omp_set_num_threads(1);
    kernel_omp_1d_row(image_1);
    verify_output(image_s, image_1, "1D Row");
    kernel_omp_1d_col(image_2);
    verify_output(image_s, image_2, "1D Col");
    kernel_omp_2d_row_block(image_3);
    verify_output(image_s, image_3, "2D Row-block");
    kernel_omp_2d_col_block(image_4);
    verify_output(image_s, image_4, "2D Col-block");
    kernel_omp_for(image_5);
    verify_output(image_s, image_5, "OMP for");

    /* Parallel run */
    for (int i = 0; i < num_configs; i++)
    {
        int t = thread_counts[i];
        omp_set_num_threads(t);
        cout << "\n--- Threads: " << t << " ---" << endl;

        start = omp_get_wtime();
        kernel_omp_1d_row(image_1);
        t1 = omp_get_wtime() - start;

        start = omp_get_wtime();
        kernel_omp_1d_col(image_2);
        t2 = omp_get_wtime() - start;

        start = omp_get_wtime();
        kernel_omp_2d_row_block(image_3);
        t3 = omp_get_wtime() - start;

        start = omp_get_wtime();
        kernel_omp_2d_col_block(image_4);
        t4 = omp_get_wtime() - start;

        start = omp_get_wtime();
        kernel_omp_for(image_5);
        t5 = omp_get_wtime() - start;

        cout << "1D Row speedup:       " << t_serial / t1 << endl;
        cout << "1D Col speedup:       " << t_serial / t2 << endl;
        cout << "2D Row-block speedup: " << t_serial / t3 << endl;
        cout << "2D Col-block speedup: " << t_serial / t4 << endl;
        cout << "OMP for speedup:      " << t_serial / t5 << endl;
    }

    /* Save result */
    save_ppm("output/fractal_serial.ppm", image_s, DIM, DIM);
    save_ppm("output/fractal_1d_row.ppm", image_1, DIM, DIM);
    save_ppm("output/fractal_1d_col.ppm", image_2, DIM, DIM);
    save_ppm("output/fractal_2d_row_block.ppm", image_3, DIM, DIM);
    save_ppm("output/fractal_2d_col_block.ppm", image_4, DIM, DIM);
    save_ppm("output/fractal_omp_for.ppm", image_5, DIM, DIM);

    delete[] image_s;
    delete[] image_1;
    delete[] image_2;
    delete[] image_3;
    delete[] image_4;
    delete[] image_5;
    return 0;
}