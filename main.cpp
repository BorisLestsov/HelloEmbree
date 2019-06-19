#include <omp.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits>
#include <iomanip>

#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

#include "vector3.h"
#include "ray.h"
#include "bxdf.h"
#include "primitive.h"
#include "utils.h"
#include "argparse.h"

#include <OpenImageDenoise/oidn.hpp>
// #include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

ArgumentParser parser("Render");

// Initial scene representation
std::vector<GeometricPrimitive*> scene_geometry;

// Camera settings
struct Camera{
    Vec3 origin;
    Vec3 dir;
    Ray  center_ray;
    Vec3 x_axis;
    Vec3 y_axis ;

    int width;
    int height;
};

// TODO(?): Implement at least stochastic sampling or stratified sampling (maybe something more complex? filtering? quasi-mc?)
// TODO(?): You may also want to implement a proper camera class.
// Generation of the initial sample
inline Vec3 generate_sample(int pixel_x, int pixel_y, int filter_dx, int filter_dy, Camera camera, bool rand=true){
    float dx, dy;
    if (rand){
        dx = (get_uniform()-0.5)*1;
        dy = (get_uniform()-0.5)*1;
    } else {
        dx = 0.5;
        dy = 0.5;
    }

    Vec3 d = camera.x_axis * ((float)(pixel_x + dx) / camera.width - 0.5) +
             camera.y_axis * ((float)(pixel_y + dy) / camera.height - 0.5) + camera.center_ray.dir;
    return normalize(d);
}


// ALBEDO
Vec3 albedo(const RTCScene& embree_scene, const Ray &ray, int max_depth) {

    Ray normal_ray = ray;
    RTCRayHit rtc_ray;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    Ray_to_RTCRayHit(normal_ray, rtc_ray);

    int id = -1;
    rtcIntersect1(embree_scene, &context, &rtc_ray);

    if (rtc_ray.hit.geomID == (int) RTC_INVALID_GEOMETRY_ID) {
        return Vec3(0.0);
    }

    id = rtc_ray.hit.geomID;
    const GeometricPrimitive* shape = scene_geometry[id];

    Vec3 p = normal_ray(rtc_ray.ray.tfar);

    Vec3 normal;
    Vec3 flipped_normal;

    if(shape->if_interpolated) {
        float inter_normal[] = {0.0f, 0.0f, 0.0f};
        rtcInterpolate0(rtcGetGeometry(embree_scene, id), rtc_ray.hit.primID, rtc_ray.hit.u, rtc_ray.hit.v,
                        RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, inter_normal, 3);
        normal = normalize(Vec3f(inter_normal));
        flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
    }
    else {
        normal = normalize(Vec3(rtc_ray.hit.Ng_x, rtc_ray.hit.Ng_y, rtc_ray.hit.Ng_z));
        flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
    }

    // Adding the emmission of the hit primitive (L_e in the rendering equation)
    for (int i = 0; i < shape->mats.size(); i++){
        MatBase* mat = shape->mats[i];
        if (mat->bxdf_type == BxDF_TYPE ::REFRACTIVE) return Vec3(1.0);
        if (mat->bxdf_type == BxDF_TYPE ::FRESNEL) return Vec3(1.0);
    }
    for (int i = 0; i < shape->mats.size(); i++){
        MatBase* mat = shape->mats[i];
        if (mat->bxdf_type == BxDF_TYPE ::DIFFUSE) return mat->color;
    }

    return Vec3(1.0);
}
// NORMALS
Vec3 get_normals(const RTCScene& embree_scene, const Ray &ray, int max_depth) {

    Ray normal_ray = ray;
    RTCRayHit rtc_ray;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    Ray_to_RTCRayHit(normal_ray, rtc_ray);

    int id = -1;
    rtcIntersect1(embree_scene, &context, &rtc_ray);

    if (rtc_ray.hit.geomID == (int) RTC_INVALID_GEOMETRY_ID) {
        return Vec3(0.0);
    }

    id = rtc_ray.hit.geomID;
    const GeometricPrimitive* shape = scene_geometry[id];

    Vec3 p = normal_ray(rtc_ray.ray.tfar);

    Vec3 normal;
    Vec3 flipped_normal;

    if(shape->if_interpolated) {
        float inter_normal[] = {0.0f, 0.0f, 0.0f};
        rtcInterpolate0(rtcGetGeometry(embree_scene, id), rtc_ray.hit.primID, rtc_ray.hit.u, rtc_ray.hit.v,
                        RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, inter_normal, 3);
        normal = normalize(Vec3f(inter_normal));
        flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
    }
    else {
        normal = normalize(Vec3(rtc_ray.hit.Ng_x, rtc_ray.hit.Ng_y, rtc_ray.hit.Ng_z));
        flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
    }

    return normal;
}


// The main part that constructs light paths and actually solves the rendering equation
Vec3 integrate(RTCScene& embree_scene, const Ray &ray, int max_depth) {

    Ray normal_ray = ray;
    RTCRayHit rtc_ray;

    // The final radiance (the left part of the rendering equation, L_o)
    Vec3 L(0.0);
    // The radiance from the current path segment (L_i in the rendering equation)
    Vec3 F(1.0);

    double prob = 0.0;

    // Creating intersection context
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    for (int depth = 0; depth < max_depth; ++depth) {
        // Setting RTCRay structure from the current Ray
        Ray_to_RTCRayHit(normal_ray, rtc_ray);

        int id = -1;
        // Test it for intersection with the scene
        rtcIntersect1(embree_scene, &context, &rtc_ray);

        // If the ray doesn't hit anything, return the default value (usually Vec3(0.0, 0.0, 0.0)) for the current hit
        // (meaning we stop tracing the path, but it still can carry the energy, depending on your implementation)
        if (rtc_ray.hit.geomID == (int) RTC_INVALID_GEOMETRY_ID) {
            return L + Vec3(0.0);
        }

        // The hit occured and we get the primitive's Id
        id = rtc_ray.hit.geomID;
        // Get the instance from the Id
        GeometricPrimitive* shape = scene_geometry[id];

        // Since we can only get the distance from the hit, calculate the actual hit point
        Vec3 p = normal_ray(rtc_ray.ray.tfar);

        Vec3 normal;
        Vec3 flipped_normal;

        // We interpolate normals to get a smooth shading (a.k.a. Gouraud shading)
        // You can set this flag in the mesh's constructor
        if(shape->if_interpolated) {
            float inter_normal[] = {0.0f, 0.0f, 0.0f};
            rtcInterpolate0(rtcGetGeometry(embree_scene, id), rtc_ray.hit.primID, rtc_ray.hit.u, rtc_ray.hit.v,
                            RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, inter_normal, 3);
            normal = normalize(Vec3f(inter_normal));
            flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
        }
        else {
            normal = normalize(Vec3(rtc_ray.hit.Ng_x, rtc_ray.hit.Ng_y, rtc_ray.hit.Ng_z));
            flipped_normal = dot(normal, normal_ray.dir) < 0 ? normal : -normal;
        }


        // TODO(?): Implement the Russian roulette here
        // Vec3 col = normalize(F);
        // prob = std::max(col.x, std::max(col.y, col.z));
        // prob = 1.0/(0.05*depth+1);
        // // prob = 0.8;
        // if (get_uniform() > prob) {
        //     // print(depth, prob);
        //     // print(depth, prob);
        //     // return L * 1.0/(1.0-prob);
        //     return Vec3(0.0);
        //     // break;
        // }
        // L *= 1.0/(prob);

        std::vector<double> psums{0};
        for (int i = 0; i < shape->mats.size(); i++){
            MatBase* mat = shape->mats[i];
            psums.push_back(psums.back() + mat->weight);
        }

        int mat_ind = 0;
        double mat_flt = get_uniform()*psums.back();
        for (int i = 0; i < shape->mats.size(); i++){
            if ((mat_flt >= psums[i]) and (mat_flt < psums[i+1])){
                mat_ind = i;
                break;
            }
        }

        MatBase* mat = shape->mats[mat_ind];
        normal_ray = mat->sample(normal_ray, p, normal, flipped_normal);

        // Adding the emmission of the hit primitive (L_e in the rendering equation)
        L += F * mat->emission;
        F *= mat->color;

    }
    // return L * 1.0/(prob);
    return L;
}


void render(RTCScene& embree_scene, const Camera& camera, const int spp_total, const int max_path_depth, const int stages){
    // Image vector
    Vec3* c = new Vec3[camera.width * camera.height];
    Vec3* c_tmp = new Vec3[camera.width * camera.height];
    Vec3* c_ldr = new Vec3[camera.width * camera.height];
    float* in_arr = new float[camera.width * camera.height*3];
    float* in_arr_hdr = new float[camera.width * camera.height*3];
    float* out_arr = new float[camera.width * camera.height*3];
    float* alb_arr = new float[camera.width * camera.height*3];
    float* n_arr = new float[camera.width * camera.height*3];

    cv::namedWindow("Render", cv::WINDOW_AUTOSIZE );

    // Iterate over all the pixels, first height, then width
    int tot_its = stages;
    int spp = spp_total/tot_its;
    for (int it = 1; it <= tot_its; it++){

#pragma omp parallel for
        for (int y = 0; y < camera.height; y++){

            if (omp_get_thread_num() == 0) {
              fprintf(stderr,"\rRendering %d stage (%d spp) %5.2f%%", it, spp,100.0*y*omp_get_max_threads()/(camera.height-1));
            }
            // TODO: You have to parallelize your renderer
            // Easy option: just use OpenMP / Intel TBB to run the for loop in parallel (don't forget to avoid writing to the same pixel at the same time by different processes)
            // Not so easy option: run ray-intersection workloads in parallel
            for (int x = 0; x < camera.width; ++x) {
                // Getting pixel's index
                int current_idx = (camera.height - y - 1) * camera.width + x;

                Vec3 r = Vec3(0.0);
                for (int s = 0; s < spp; s++) {
                // Generate direction for the initial ray
                    Vec3 d = generate_sample(x, y, 0.0, 0.0, camera);
                    // Add light path's contribution to the pixel
                    Vec3 tmp = integrate(embree_scene, Ray(camera.center_ray.org, d), max_path_depth);
                    r = r +  tmp * (1.0 / spp);
                }
                // You might want to clamp the values in the end
                c[current_idx] = c[current_idx] + Vec3(r.x, r.y, r.z);
            }
        }
#pragma omp parallel for
        for (int y = 0; y < camera.height; y++){
            for (int x = 0; x < camera.width; ++x) {
                int current_idx = (camera.height - y - 1) * camera.width + x;
                c[current_idx] = c[current_idx];
                c_tmp[current_idx] = clamp(c[current_idx]) / (double) it;

                in_arr[3*current_idx+0] = c_tmp[current_idx].x;
                in_arr[3*current_idx+1] = c_tmp[current_idx].y;
                in_arr[3*current_idx+2] = c_tmp[current_idx].z;

            }
        }
        cv::Mat cv_img = cv::Mat(camera.height, camera.width, CV_32FC3, in_arr);
        cv::cvtColor(cv_img, cv_img, CV_BGR2RGB);

        save_ppm(camera.width, camera.height, c_tmp, std::to_string(it) + std::string("_") + std::to_string(spp));
        cv::imshow("Render", cv_img);
        cv::waitKey(1000);
    }

    // float mean = 0.0;
    // for (int y = 0; y < camera.height; y++){
    //     for (int x = 0; x < camera.width; ++x) {
    //         int current_idx = (camera.height - y - 1) * camera.width + x;
    //         mean += c[current_idx].x + c[current_idx].y + c[current_idx].z;
    //     }
    // }
    // mean /= camera.height*camera.width;

    // cv::destroyWindow("Render");
#pragma omp parallel for
    for (int y = 0; y < camera.height; y++){
        if (omp_get_thread_num() == 0) {
           fprintf(stderr,"\rPostprocessing (%d spp) %5.2f%%",spp,100.0*y*omp_get_max_threads()/(camera.height-1));
        }
        for (int x = 0; x < camera.width; ++x) {
            int current_idx = (camera.height - y - 1) * camera.width + x;

            c_ldr[current_idx] = clamp(c[current_idx] / (double) tot_its);

            in_arr_hdr[3*current_idx+0] = c[current_idx].x;
            in_arr_hdr[3*current_idx+1] = c[current_idx].y;
            in_arr_hdr[3*current_idx+2] = c[current_idx].z;

            in_arr[3*current_idx+0] = c_ldr[current_idx].x;
            in_arr[3*current_idx+1] = c_ldr[current_idx].y;
            in_arr[3*current_idx+2] = c_ldr[current_idx].z;

            Vec3 d = generate_sample(x, y, 0.0, 0.0, camera, false);
            Vec3 a = albedo(embree_scene, Ray(camera.center_ray.org, d), max_path_depth);
            alb_arr[3*current_idx+0] = a.x;
            alb_arr[3*current_idx+1] = a.y;
            alb_arr[3*current_idx+2] = a.z;

            Vec3 n = get_normals(embree_scene, Ray(camera.center_ray.org, d), max_path_depth);
            n_arr[3*current_idx+0] = n.x;
            n_arr[3*current_idx+1] = n.y;
            n_arr[3*current_idx+2] = n.z;

        }
    }

    cv::Mat cv_img_hdr = cv::Mat(camera.height, camera.width, CV_32FC3, in_arr_hdr);
    cv::cvtColor(cv_img_hdr, cv_img_hdr, CV_BGR2RGB);
    cv::imwrite("hdr.exr", cv_img_hdr);


    save_ppm(camera.width, camera.height, c_ldr, "final_ldr");

    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  in_arr,  oidn::Format::Float3, camera.width, camera.height);
    filter.setImage("output", out_arr, oidn::Format::Float3, camera.width, camera.height);
    filter.setImage("albedo", alb_arr, oidn::Format::Float3, camera.width, camera.height);
    filter.setImage("normal", n_arr, oidn::Format::Float3, camera.width, camera.height);
    filter.commit();

    // Filter the image
    filter.execute();

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
      std::cout << "Error: " << errorMessage << std::endl;


#pragma omp parallel for
    for (int y = 0; y < camera.height; y++){
        for (int x = 0; x < camera.width; ++x) {
            int current_idx = (camera.height - y - 1) * camera.width + x;
            c[current_idx].x = out_arr[3*current_idx+0];
            c[current_idx].y = out_arr[3*current_idx+1];
            c[current_idx].z = out_arr[3*current_idx+2];

            n_arr[3*current_idx+0] = std::abs(n_arr[3*current_idx+0]);
            n_arr[3*current_idx+1] = std::abs(n_arr[3*current_idx+1]);
            n_arr[3*current_idx+2] = std::abs(n_arr[3*current_idx+2]);
        }
    }
    save_ppm(camera.width, camera.height, c, "denoised");

    cv::Mat cv_img = cv::Mat(camera.height, camera.width, CV_32FC3, in_arr);
    cv::Mat cv_img_denoise = cv::Mat(camera.height, camera.width, CV_32FC3, out_arr);
    cv::Mat alb_img = cv::Mat(camera.height, camera.width, CV_32FC3, alb_arr);
    cv::Mat n_img = cv::Mat(camera.height, camera.width, CV_32FC3, n_arr);

    cv::Mat res;

    cv::hconcat(std::vector<cv::Mat>{cv_img, cv_img_denoise, alb_img, n_img}, res);
    cv::cvtColor(res, res, CV_BGR2RGB);

    cv::imshow("Render", res);
    cv::waitKey(0);


    delete [] c;
    delete [] in_arr;
    delete [] in_arr_hdr;
    delete [] out_arr;
    delete [] alb_arr;
    delete [] n_arr;
}

int main(int argc, char *argv[]){

    parser.add_argument("-h", "--height", "", true);
    parser.add_argument("-w", "--width", "", true);
    parser.add_argument("-s", "--spp", "", true);
    parser.add_argument("-p", "--stages", "", true);
    parser.add_argument("-d", "--max-depth", "", true);

    parser.parse(argc, argv);

    std::cout << std::setprecision(16);

    // Image resolution, SPP
    int film_width        = parser.get<int>("width");
    int film_height       = parser.get<int>("height");
    int samples_per_pixel = parser.get<int>("spp");
    int max_depth = parser.get<int>("d");
    int stages = parser.get<int>("stages");

    // Setting the camera
    Camera camera_desc;
    camera_desc.origin        = Vec3(0.0, 0.0, -120.0);
    camera_desc.dir           = Vec3(0.0, 0.0, 1.0);
    camera_desc.center_ray    = Ray(camera_desc.origin, normalize(camera_desc.dir));
    camera_desc.x_axis        = Vec3(1.0, 0.0, 0.0);
    camera_desc.y_axis        = normalize(cross(camera_desc.x_axis, -camera_desc.center_ray.dir));
    camera_desc.width         = film_width;
    camera_desc.height        = film_height;


    // Path to our models
    std::string dragon  = "../assets/dragon.obj";
    std::string bunny   = "../assets/bunny.obj";

    // Zero transform and transforms for our models
    Transform zero_trans    = Transform(Vec3(1.0, 1.0, 1.0), Vec3(0.0), Vec3(0.0));
    Transform dragon_trans  = Transform(Vec3(70.0, 70.0, 70.0), Vec3(0.0), Vec3(0.0, -30.0, 0.0));
    //Transform bunny_trans   = Transform(Vec3(40.0, 40.0, 40.0), Vec3(0.0), Vec3(10.0, -50.0, 0.0));
    Transform bunny_trans   = Transform(Vec3(20.0, 20.0, 20.0), Vec3(0.0, 0.0, 0.0), Vec3(40.0, -60.0, 20.0));

    const double        r = 10000.0;
    const double offset_r = 10050.0;

    // Cornell box
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3(-offset_r, 0.0, 0.0),
                             MatVec{new MatDiffuse(Vec3(0.75,0.25,0.25), Vec3(0.0))}
    )));//Left
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3( offset_r, 0.0, 0.0),
                             MatVec{new MatDiffuse(Vec3(0.25,0.25,0.75), Vec3(0.0))}
    )));//Right
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3(0.0, 0.0, -offset_r - 100.0),
                             MatVec{new MatDiffuse(Vec3(0.75,0.75,0.75), Vec3(0.0))}
    )));//Back
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3(0.0, 0.0,  offset_r),
                             MatVec{new MatDiffuse(Vec3(0.75,0.75,0.75), Vec3(0.0))}
    )));//Front
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3(0.0, -offset_r, 0.0),
                             MatVec{new MatDiffuse(Vec3(0.75,0.75,0.75), Vec3(0.0))}
    )));//Bottom
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(r,      zero_trans, Vec3(0.0,  offset_r, 0.0),
                             MatVec{new MatDiffuse(Vec3(0.75,0.75,0.75), Vec3(0.0))}
    )));//Top
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(5000.0, zero_trans, Vec3(0.0, 5049.99, 0.0),
                             // MatVec{new MatDiffuse(Vec3(1.0,1.0,1.0), Vec3(12.0))}
                             MatVec{new MatDiffuse(Vec3(1.0, 1.0, 0.5), Vec3(0.0), 0.2),
                                    new MatDiffuse(Vec3(0.0), Vec3(24.0), 0.8)
                             }
    )));//Light

    // Other objects
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new TriangleMesh(dragon, dragon_trans, Vec3(0.0),
                             MatVec{new MatDiffuse(Vec3(0.83, 0.68, 0.21), Vec3(0.0), 0.5),
                                    new MatGlossy(Vec3(0.83, 0.68, 0.21), Vec3(0.0), 20, 0.5)
                             }
    )));
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new TriangleMesh(bunny, bunny_trans, Vec3(0.0),
                             MatVec{new MatDiffuse(Vec3(0.0, 0.0, 0.75), Vec3(1.0), 0.2),
                                    new MatFresnel(Vec3(1.0, 1.0, 1.0), Vec3(0.0), 0.8)
                             }
    )));

    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(10.0,      zero_trans, Vec3(20.0,  10.0, 0.0),
                             MatVec{new MatSpecular(Vec3(1.0, 1.0, 1.0), Vec3(0.0))}
    )));//Top
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(15.0,      zero_trans, Vec3(-30.0,  10.0, 0.0),
                             MatVec{new MatFresnel(Vec3(1.0, 1.0, 1.0), Vec3(0.0), 1.55)}
    )));//Top
    scene_geometry.push_back(static_cast<GeometricPrimitive*>(new Sphere(10.0,      zero_trans, Vec3(30.0,  30.0, 0.0),
                             MatVec{new MatFresnel(Vec3(0.83, 0.68, 0.21), Vec3(0.0), 1.15)}
    )));//Top


    // Creating a new device
    RTCDevice rtc_device = rtcNewDevice("threads=0");

    // Creating a new scene
    RTCScene rtc_scene = rtcNewScene(rtc_device);
    //rtcSetSceneFlags(rtc_scene, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_ROBUST);

    // Constructing Embree objects, setting VBOs/IBOs
    for(int i = 0; i < scene_geometry.size(); ++i) {
        scene_geometry[i]->construct_embree_object(rtc_device, rtc_scene);
    }

    // Loading the scene
    rtcCommitScene(rtc_scene);

    // Start rendering
    render(rtc_scene, camera_desc, samples_per_pixel, max_depth, stages);

    // Releasing the scene and then the device
    rtcReleaseScene(rtc_scene);
    rtcReleaseDevice(rtc_device);

    scene_geometry.clear();
}
