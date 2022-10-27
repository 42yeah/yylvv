#ifndef YYLVV_STREAMLINE_CUH
#define YYLVV_STREAMLINE_CUH

#include "../renderstate.cuh"
#include "../nrrd/VectorField.h"
#include "../VAO.h"
#include "../Program.h"
#include <vector>

class StreamLineRenderState : public RenderState 
{
public:
    StreamLineRenderState();
    ~StreamLineRenderState();

    virtual void initialize(YYLVVRes &res, UIRes &ui_res) override;
    virtual void destroy() override;
    virtual void render(YYLVVRes &res, UIRes &ui_res) override;
    virtual void process_events(YYLVVRes &res, UIRes &ui_res) override;
    virtual void key_pressed(YYLVVRes &res, UIRes &ui_res, int key) override;

private:
    bool allocate_graphics_resources();
    bool generate_streamlines(YYLVVRes &res, UIRes &ui_res);
    bool generate_seed_points(const BBox &bbox, int num_seeds);
    bool trace_streamlines(YYLVVRes &res, UIRes &ui_res);

    bool trace_streamlines_adaptive(YYLVVRes &res, UIRes &ui_res);
    bool simplify_streamlines();

    // seed point generation configurations
    int num_seeds, num_lines;
    float simulation_dt;
    std::vector<glm::vec3> seed_points;

    // streamline variables
    std::shared_ptr<VAO> streamline_vao;
    std::shared_ptr<Program> streamline_program;
    cudaGraphicsResource *streamline_graphics_resource;
    bool use_runge_kutta_4_integrator;
    float seeding_plane_x;
    bool adaptive_mode;

    // adaptive mode streamline variables
    int num_seeds_adaptive;
    float adaptive_explosion_radius;
    int num_explosion;
    int explosion_cooldown_counter;

    // streamline simplification
    bool do_simplify;
};

#endif