#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <signal.h>
#include <exception>

#define X_RES 2048
#define Y_RES 2048

__device__ float ray_sphere_intersect(float cx, float cy, float cz, float clipx, float clipy, float clipz, float spherex, float spherey, float spherez, float spherer)
{
    float a = clipx*clipx+clipy*clipy+clipz*clipz;
    float b = 2.0f*(clipx*(cx-spherex)+clipy*(cy-spherey)+clipz*(cz-spherez));
    float c = cx*cx+spherex*spherex+cy*cy+spherey*spherey+cz*cz+spherez*spherez-2.0f*cx*spherex-2.0f*cy*spherey-2.0f*cz*spherez-spherer*spherer;
    float discriminant = b*b-4.0f*a*c;
    if (discriminant < 0) {
        return 0.0f;
    }
    else {
        return (-b - sqrtf(discriminant)) / (2.0f*a);
    }
}

__device__ float unit_x(float x, float y, float z)
{
    float mag = sqrtf(x*x+y*y+z*z);
    return x/mag;
}

__device__ float unit_y(float x, float y, float z)
{
    float mag = sqrtf(x*x+y*y+z*z);
    return y/mag;
}

__device__ float unit_z(float x, float y, float z)
{
    float mag = sqrtf(x*x+y*y+z*z);
    return z/mag;
}

__device__ float clamp(float x, float a, float b)
{
    return fmaxf(a, fminf(b, x));
}

__global__ void stress(float *a, float *objects, int numobjects)
{
    while (true)
    {
        int idx = ((blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x*gridDim.x+(blockIdx.x*blockDim.x+threadIdx.x))*3;

        float numxsamples = 20.0f;
        float numysamples = 20.0f;
        float numsamples = numxsamples*numysamples;

        float cx = 0.0f;
        float cy = 0.0f;
        float cz = 1.0f;

        float lightx = 10.0f;
        float lighty = 6.0f;
        float lightz = 0.0f;
        float lightint = 1.0f; // light intensity

        for (int xs = 0; xs<=numxsamples; xs++) {
            for (int ys = 0; ys<=numysamples; ys++) {
                float clipx = (blockIdx.x*blockDim.x+threadIdx.x)/16.0f - 8.0f + (xs-numxsamples/2.0f)*0.1f/numxsamples;
                float clipy = (blockIdx.y*blockDim.y+threadIdx.y)/16.0f - 8.0f + (ys-numysamples/2.0f)*0.1f/numysamples;
                float clipz = -50.0f; // clipping plane distance

                // Ray Coloring
                float r = 1.0f-blockIdx.y/512.0f;
                float g = 1.0f-blockIdx.y/512.0f;
                float b = 1.0f;

                float sphere_dist;
                float spherex, spherey, spherez, spherer, sphere_col_r, sphere_col_g, sphere_col_b;

                bool processed = false;

                for (int i = 0; i < numobjects; i++) {
                
                    spherex = objects[i*7];
                    spherey = objects[i*7+1];
                    spherez = objects[i*7+2];
                    spherer = objects[i*7+3];
                    sphere_col_r = objects[i*7+4];
                    sphere_col_g = objects[i*7+5];
                    sphere_col_b = objects[i*7+6];

                    float sd = ray_sphere_intersect(cx, cy, cz, clipx, clipy, clipz, spherex, spherey, spherez, spherer);

                    if (0.0f < sd) {
                        if (!processed) { // pixel still empty?
                            sphere_dist = sd;
                            processed = true;
                        }
                        if (sd <= sphere_dist) {
                            sphere_dist = sd;

                            // Calculate surface normal <n>
                            float nx = cx+sphere_dist*clipx-spherex;
                            float ny = cy+sphere_dist*clipy-spherey;
                            float nz = cz+sphere_dist*clipz-spherez;
                            nx = unit_x(nx, ny, nz);
                            ny = unit_y(nx, ny, nz);
                            nz = unit_z(nx, ny, nz);

                            float tlx = lightx-nx; // from point to light x
                            float tly = lighty-ny;
                            float tlz = lightz-nz;
                            tlx = unit_x(tlx, tly, tlz);
                            tly = unit_y(tlx, tly, tlz);
                            tlz = unit_z(tlx, tly, tlz);

                            float light_dot = ((nx*tlx+ny*tly+nz*tlz-1.0f)/-2.0f)*lightint;

                            r = sphere_col_r*light_dot;
                            g = sphere_col_g*light_dot;
                            b = sphere_col_b*light_dot;
                        }
                    }
                }
            a[idx] += r;
            a[idx+1] += g;
            a[idx+2] += b;
            }
        }
        a[idx] = clamp(a[idx]/numsamples, 0.0f, 1.0f)*255.0f;
        a[idx+1] = clamp(a[idx+1]/numsamples, 0.0f, 1.0f)*255.0f;
        a[idx+2] = clamp(a[idx+2]/numsamples, 0.0f, 1.0f)*255.0f;
    }
}

int main() {

    printf("Setting up...\n");

    int cudaDeviceCount = 0;

    cudaGetDeviceCount(&cudaDeviceCount);

    printf("Located %i GPUS to use...\n", cudaDeviceCount);

    float *d_a, *d_b;
    const int len = X_RES*Y_RES*3;
    float *a = new float[len];

    float b[] = {-1.0f, 0.0f, -1.3f, 0.8f, 0.8f,  0.8f,  0.75f,
                0.0f,  -0.8f, -1.3f, 1.0f, 0.64f, 0.11f, 0.08f,
                1.0f,   0.0f, -1.3f, 0.8f, 0.8f,  0.8f,  0.75f};

    int numobjects = 3;

    dim3 numBlocks(256, 256, 1);
    dim3 threadsPerBlock(32, 32, 1);

    cudaMalloc((void**)&d_a, len*sizeof(float));
    cudaMalloc((void**)&d_b, numobjects*7*sizeof(float));

    cudaMemcpy(d_a, a, len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, numobjects*7*sizeof(float), cudaMemcpyHostToDevice);

    printf("Starting D2O GPU Test! (Press ctrl+c to terminate)\n");

    for (int i = 0; i<cudaDeviceCount; i++) { // Stress test for multiple GPUS currently unsupported
        cudaSetDevice(i);
        printf("Starting GPU Test on GPU %i\n", i);
        stress<<<numBlocks, threadsPerBlock>>>(d_a, d_b, numobjects);
    }
    cudaDeviceSynchronize();

    // we're never gonna get here :/
    cudaMemcpy(a, d_a, len*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

    free(a);
    free(b);

    printf("Finished D2O GPU Test!\n");

    return 0;
}