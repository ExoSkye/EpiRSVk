#version 460

layout(local_size_x = 64) in;

struct Person {
    vec2 person_pos;
    uint status;
    uint infected;
};

struct Vertex {
    vec2 vertex_pos;
    vec3 color;
};

// RW Buffers
layout (set = 0, binding = 0) buffer people_buf {
    Person people[];
};

layout (set = 1, binding = 0) buffer verticies_buf {
    Vertex verticies[];
};

// Read-only info
layout (set = 2, binding = 0) uniform rodata {
    uint region_size_x;
    uint region_size_y;
    uint size_x;
    uint size_y;
    uint len;
    float seed;
};

// Stuff for random number generation

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

// End RNG code

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < len) {
        vec2 pos = people[index].person_pos;
        vec2 old_pos = pos;

        float rx = random(seed + index);
        float ry = random(seed + index + rx);

        pos.x += (rx - 0.5) * 2;
        pos.y += (ry - 0.5) * 2;

        if (pos.x > size_x) {
            pos = old_pos - vec2(2,0);
        }
        if (pos.x < 0) {
            pos = old_pos + vec2(2,0);
        }
        if (pos.y > size_x) {
            pos = old_pos - vec2(0,2);
        }
        if (pos.x < 0) {
            pos = old_pos + vec2(0,2);
        }

        people[index].person_pos = pos;


        float real_x = (pos.x / (size_x / 2)) - 1;
        float real_y = (pos.y / (size_y / 2)) - 1;

        verticies[index].vertex_pos = vec2(real_x, real_y);
        verticies[index].color = vec3(1.0, 1.0, 1.0);
    }
}
