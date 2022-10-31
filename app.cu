#include "app.cuh"
#include <glm/gtc/type_ptr.hpp>

// For GLFW stuffs
App *bound_app = nullptr;

App::App(YYLVVRes &res) : res(res),
    window(res.window),
    valid(false)
{
    if (!init())
    {
        std::cerr << "Failed to initialize UI & its resources?" << std::endl;
        return;
    }
    valid = true;
}

App::~App()
{
    if (render_state) {
        render_state->destroy();
    }
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(ctf_tex_cuda));
    CHECK_CUDA_ERROR(cudaFreeArray(ctf_data_cuda));
}

bool App::init()
{
    std::cout << "Initializing bounding box and bounding box program." << std::endl;
    const glm::vec3 a = glm::vec3(-0.5f, -0.5f, -0.5f),
                b = glm::vec3(0.5f, -0.5f, -0.5f),
                c = glm::vec3(0.5f, -0.5f, 0.5f),
                d = glm::vec3(-0.5f, -0.5f, 0.5f),
                e = a + glm::vec3(0.0f, 1.0f, 0.0f),
                f = b + glm::vec3(0.0f, 1.0f, 0.0f),
                g = c + glm::vec3(0.0f, 1.0f, 0.0f),
                h = d + glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 bounding_box_data[] = 
    {
            a, b, b, c, c, d, d, a,
            e, f, f, g, g, h, h, e,
            a, b, b, f, f, e, e, a,
            d, c, c, g, g, h, h, d,
            a, d, d, h, h, e, e, a,
            b, c, c, g, g, f, f, b
    };

    bounding_box_vao = VAO::make_vao(bounding_box_data,
                                     sizeof(bounding_box_data),
                                     GL_STATIC_DRAW,
                                     {VertexAttribPointer(0,
                                                          3,
                                                          GL_FLOAT,
                                                          GL_FALSE,
                                                          sizeof(float) * 3,
                                                          nullptr)},
                                     GLDrawCall(GL_LINES, 0, 48));

    std::cout << "Compiling line drawing program." << std::endl;
    bounding_box_program = Program::make_program("shaders/lines.vert", "shaders/lines.frag");
    
    if (!bounding_box_program || !bounding_box_program->valid) 
    {
        std::cerr << "Cannot link line-drawing program?" << std::endl;
        return false;
    }

    std::cout << "Configuring OpenGL & GLFW." << std::endl;
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetKeyCallback(window, key_callback_glfw);
    glfwSetCursorPosCallback(window, cursor_pos_callback_glfw);
    render_state = nullptr;
    glfwGetFramebufferSize(window, &screen_width, &screen_height);
    glfwSwapInterval(1);
    last_instant = glfwGetTime();
    delta_time = 0.0f;

    // Initialize delta wing
    std::cout << "Evaluating delta wing bounding box and allocating graphics resources." << std::endl;
    delta_wing_bounding_box = res.vf_tex.get_bounding_box();
    float x_min = 50.0f;
    float x_max = 169.0f;
    float y_min = 27.5f;
    float y_mid = 100.0f;
    float y_max = 172.5f;
    float z = 0.0f;
    std::vector<float3> delta_wing_fl3 = 
    {
        make_float3(x_min, y_mid, z),
        make_float3(x_max, y_min, z),
        make_float3(x_max, y_max, z)
    };
    std::cout << "Compiling delta wing shader." << std::endl;
    delta_wing_vao = VAO::make_vao(delta_wing_fl3.data(), delta_wing_fl3.size() * sizeof(float3), GL_STATIC_DRAW,
                                   {
                                       VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr)
                                   },
                                   GLDrawCall(GL_TRIANGLES, 0, 3));
    delta_wing_program = Program::make_program("shaders/lines.vert", "shaders/delta.frag");
    
    if (!delta_wing_program || !delta_wing_program->valid) 
    {
        return false;
    }

    // Color transfer function (CTF)
    std::vector<float4> vector_magnitude_ctf;
    std::cout << "Initializing color transfer function for CUDA: creating device array." << std::endl;
    vector_magnitude_ctf.push_back(make_float4(0.4f, 0.6f, 0.9f, 1.0f)); // TODO: 1D texture is weird because it has a
    vector_magnitude_ctf.push_back(make_float4(0.0f, 1.0f, 0.0f, 1.0f)); //       weird padding of 0.5
    vector_magnitude_ctf.push_back(make_float4(0.9f, 0.9f, 0.0f, 1.0f));
    vector_magnitude_ctf.push_back(make_float4(1.0f, 0.0f, 0.0f, 1.0f));
    int vm_size_in_bytes = vector_magnitude_ctf.size() * sizeof(float4);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CHECK_CUDA_ERROR(cudaMallocArray(&ctf_data_cuda, &desc, vector_magnitude_ctf.size(), 0, 0));
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(ctf_data_cuda, 0, 0, vector_magnitude_ctf.data(), vm_size_in_bytes, vm_size_in_bytes, 1, cudaMemcpyHostToDevice));

    std::cout << "Creating vector magnitude CTF texture." << std::endl;
    cudaResourceDesc rdesc;
    std::memset(&rdesc, 0, sizeof(cudaResourceDesc));
    {
        rdesc.resType = cudaResourceTypeArray;
        rdesc.res.array.array = ctf_data_cuda;
    }
    cudaTextureDesc tdesc;
    std::memset(&tdesc, 0, sizeof(cudaTextureDesc));
    {
        tdesc.addressMode[0] = cudaAddressModeWrap;
        tdesc.filterMode = cudaFilterModeLinear;
        tdesc.readMode = cudaReadModeElementType;
        tdesc.normalizedCoords = 0; // let's try normalizing it
    }
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&ctf_tex_cuda, &rdesc, &tdesc, nullptr));
    std::cout << "Vector magnitude CTF texture creation complete." << std::endl;

    align_camera();

    return true;
}

void App::key_callback_glfw(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    assert(bound_app != nullptr);
    bound_app->key_callback(window, key, scancode, action, mods);
}

void App::cursor_pos_callback_glfw(GLFWwindow *window, double xpos, double ypos)
{
    assert(bound_app != nullptr);
    bound_app->cursor_pos_callback(window, xpos, ypos);
}

void App::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action != 1) {
        return;
    }
    switch (key) {
        case GLFW_KEY_L:
            switch_state(std::make_shared<LineGlyphRenderState>());
            break;

        case GLFW_KEY_G:
            switch_state(std::make_shared<ArrowGlyphRenderState>());
            break;

        case GLFW_KEY_Z:
            switch_state(std::make_shared<StreamLineRenderState>());
            break;
    }
    if (render_state) {
        render_state->key_pressed(*this, key);
    }
}

void App::cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    ypos = -ypos;
    xpos = -xpos;
    if (!camera.prev_cursor_pos) {
        camera.prev_cursor_pos = glm::dvec2(xpos, ypos);
        return;
    }
    glm::dvec2 curr_pos = glm::dvec2(xpos, ypos);
    glm::dvec2 delta_pos = curr_pos - *camera.prev_cursor_pos;
    camera.yaw += delta_pos.x * camera.sensitivity;
    camera.pitch += delta_pos.y * camera.sensitivity;
    camera.prev_cursor_pos = curr_pos;
    camera.update_components(screen_width, screen_height);
}

void App::align_camera()
{
    glm::vec3 extent = delta_wing_bounding_box.extend(); // TODO: a typo
    float max_ext = glm::max(glm::max(extent.x, extent.y), extent.z);
    float init_dist = glm::max(extent.x, extent.y) * 0.5f * sqrt(3.0f);
    camera.eye = delta_wing_bounding_box.center() - glm::vec3(0.0f, 0.0f, init_dist);
    camera.speed = max_ext * 0.1f; // whole thing in 10 seconds
    camera.z_near = 1.0f;
    camera.z_far = max_ext * 2.0f + fabs(init_dist);
    camera.update_components(screen_width, screen_height);
}

void App::handle_continuous_key_events()
{
    if (glfwGetKey(window, GLFW_KEY_W)) 
    {
        camera.eye += camera.front * camera.speed * delta_time;
        camera.update_components(screen_width, screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_S)) 
    {
        camera.eye -= camera.front * camera.speed * delta_time;
        camera.update_components(screen_width, screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_A)) 
    {
        camera.eye -= camera.right * camera.speed * delta_time;
        camera.update_components(screen_width, screen_height);
    }
    if (glfwGetKey(res.window, GLFW_KEY_D)) 
    {
        camera.eye += camera.right * camera.speed * delta_time;
        camera.update_components(screen_width, screen_height);
    }
}

void App::loop()
{
    while (!glfwWindowShouldClose(window)) 
    {
        bound_app = this;
        glfwPollEvents();

        double this_instant = glfwGetTime();
        delta_time = (float) (this_instant - last_instant);
        last_instant = this_instant;

        handle_continuous_key_events();

        if (render_state) 
        {
            render_state->process_events(*this);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        draw_delta_wing();

        if (render_state) 
        {
            render_state->render(*this);
        }

        glfwSwapBuffers(window);
    }
}

void App::draw_delta_wing() const
{
    // 1. Draw the bounding box (that we calculated)
    bounding_box_program->use();
    glm::mat4 model = glm::translate(glm::mat4(1.0f), delta_wing_bounding_box.center());
    model = glm::scale(model, delta_wing_bounding_box.extend());
    glUniformMatrix4fv(bounding_box_program->at("model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(bounding_box_program->at("view"), 1, GL_FALSE, glm::value_ptr(camera.view));
    glUniformMatrix4fv(bounding_box_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(camera.perspective));
    bounding_box_vao->draw();

    // 2. Draw the delta wing triangle
    delta_wing_program->use();
    glUniformMatrix4fv(delta_wing_program->at("model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniformMatrix4fv(delta_wing_program->at("view"), 1, GL_FALSE, glm::value_ptr(camera.view));
    glUniformMatrix4fv(delta_wing_program->at("perspective"), 1, GL_FALSE, glm::value_ptr(camera.perspective));
    delta_wing_vao->draw();
}

void App::switch_state(std::shared_ptr<RenderState> new_state)
{
    if (render_state != nullptr) {
        render_state->destroy();
    }
    render_state = new_state;
    render_state->initialize(*this);
}
