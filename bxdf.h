#pragma once

#include "vector3.h"
#include "utils.h"
#include <cmath>

enum BxDF_TYPE {
    DIFFUSE,
    SPECULAR,
    REFRACTIVE
};


// Ideal specular reflection and it's discreet probability
// 'direction' is oriented TO the surface
// in other renderers (e.g. Mitsuba/pbrt) it is the opposite (= '2*n*dot(d,n) - d')
inline Vec3 reflect_at_normal(const Vec3 &dir, const Vec3 &n){
    return dir - n * 2.0 * dot(n, dir);
}

// Just a 'typedef' for convenience
// 'direction' is oriented AWAY FROM the surface
inline Vec3 specular_reflection(const Vec3 &wi, const Vec3 &n){
    return reflect_at_normal(-wi, n);
}

// Uniform sample on the sphere and it's PDF
inline Vec3 uniform_sphere(const double rnd1, const double rnd2){
    const double cos_theta = 1.0 - 2.0 * rnd1;
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    const double phi = 2.0 * M_PI * rnd2;
    Vec3 res_vec = Vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    return res_vec;
}

inline double uniform_sphere_pdf() {
    return D_INV4_PI;
}

// Uniform sample on the hemisphere and it's PDF
inline Vec3 uniform_hemisphere(const double rnd1, const double rnd2){
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - rnd1 * rnd1));
    const double phi = 2.0 * M_PI * rnd2;
    Vec3 res_vec = Vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, rnd2);

    return res_vec;
}

inline double uniform_hemisphere_pdf() {
    return D_INV2_PI;
}

// Uniform sample on the cosine weighted hemisphere and it's PDF
inline Vec3 cosine_weighted_hemisphere(const double rnd1, const double rnd2){
    const double cos_theta = std::sqrt(1.0 - rnd1);
    const double sin_theta = std::sqrt(rnd1);
    const double phi = 2.0 * M_PI * rnd2;
    return Vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

inline double cosine_wighted_hemisphere_pdf(double cos_theta) {
    return cos_theta * D_INV_PI;
}

inline double cosine_wighted_hemisphere_pdf(const Vec3 wi, const Vec3 n, const Vec3 wo){
    double cos_theta = std::abs(dot(wo, n));
    return cos_theta * D_INV_PI;
}

inline Vec3 cart2sph(const Vec3 n){
    auto r_n = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    auto th_n = std::acos(n.z/r_n);
    auto phi_n = std::atan2(n.y,n.x);
    return Vec3(r_n, th_n, phi_n);
}

inline Vec3 sph2cart(const Vec3 v){
    Vec3 res = Vec3(0.0);
    res.x = v.x * std::sin(v.y) * std::cos(v.z);
    res.y = v.x * std::sin(v.y) * std::sin(v.z);
    res.z = v.x * std::cos(v.y);
    return res;

}
