import pygame as pg
import math
import numpy as np

WIDTH, HEIGHT = 800, 800
PI = 3.141592653589
coeff_of_restitution = 0.8

def vec_add(vectors):
    return [sum([v[0] for v in vectors]), sum([v[1] for v in vectors])] 

def vec_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1]]

def scalar_mul(v, s):
    return [v[0] * s, v[1] * s]

def cross(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def vector_cross(s, v):
    return [-v[1] * s, v[0] * s]

def SAT_normals(normals):
    if len(normals) % 2 != 0:
        return normals
    out = []
    for i, j in zip(normals[:int(len(normals)/2)], normals[int(len(normals)/2):]):
        if vec_add([i, j]) == [0, 0]:
            out.append(i)
    return out

def clip(points, normal, c):
    if len(points) < 2:
        return points 

    out = []
    d1 = dot(normal, points[0]) - c
    d2 = dot(normal, points[1]) - c

    if d1 <= 0:
        out.append(points[0])

    if d2 <= 0:
        out.append(points[1])

    if d1 * d2 < 0:
        alpha = d1 / (d1 - d2)
        edge = vec_sub(points[1], points[0])
        intersection = vec_add([points[0], scalar_mul(edge, alpha)])
        out.append(intersection)

    return out

def get_shoelace_area(vertices):
    n = len(vertices)

    if n < 3:
        return 0
    
    signed_area = 0.0
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]

        signed_area = (v1[0] * v2[1]) - (v2[0] * v1[1])   

    return signed_area

def anticlockwiseify(vertices):
    if len(vertices) < 3:
        return vertices

    area = get_shoelace_area(vertices)

    if area < 0:
        return vertices[::-1] 
    else:
        return vertices
    
def get_edges(points):
    edges = []
    for i in range(1, len(points)):
        edges.append((points[i-1], points[i]))
    edges.append((points[len(points) - 1], points[0]))
    return edges

def get_normals(edges):
    ns = []
    for edge in edges:
        ex = edge[1][0] - edge[0][0]
        ey = edge[1][1] - edge[0][1]

        nx = ey
        ny = -ex

        mag = math.sqrt(nx*nx + ny*ny)
        nx /= mag
        ny /= mag
        ns.append((nx, ny))

    return ns

def getAABB(vertices):
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    return [min_x, min_y, max_x, max_y]

def rotate_vector(vector, angle):
    x, y = vector

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    return [x_new, y_new]




class Vec2:
    def __init__(self, pos, components):
        self.pos = pos
        self.components = components
        self.mag = math.sqrt(components[0]**2 + components[1]**2)

class PolygonShape():
    def __init__(self, points, color=(255, 0, 0)):
        self.local_points = anticlockwiseify(points)
        self.edges = get_edges(self.local_points)
        self.points = self.local_points[:]
        self.color = color

class CircleShape:
    def __init__(self, radius, color=(255, 0, 0)):
        self.radius = radius
        self.color = color
class RigidBody:
    def __init__(self, shape, M, I, damping=1, pos=None, theta=0, static=False, rotatable=True):

        self.static = static

        self.shape = shape
        self.M = M
        self.inv_M = 1/M if not static else 0
        self.I = I
        self.inv_I = 0 if static or not rotatable else 1/I
        self.damping = damping
        self.pos = pos if pos is not None else [0, 0]
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.theta = theta * PI /180
        self.omega = 0
        self.alpha = 0
        self.forces = []
        self.torques = []
        self.angular_impulses = []
        self.impulses = []


        if isinstance(self.shape, PolygonShape):
            self.shape.points = [vec_add([rotate_vector(p, self.theta), self.pos]) for p in self.shape.local_points]
            self.shape.edges = get_edges(self.shape.points)
            self.normals = get_normals(self.shape.edges)
            self.AABB = getAABB(self.shape.points)
        elif isinstance(self.shape, CircleShape):
            r = self.shape.radius
            x, y = self.pos
            self.AABB = [x-r, y-r, x+r, y+r]

    def draw_body(self, surf, color):
        if isinstance(self.shape, PolygonShape):
            pg.draw.polygon(surf, color, self.shape.points)
        elif isinstance(self.shape, CircleShape):
            pg.draw.circle(surf, color, self.pos, self.shape.radius)
        pg.draw.circle(surf, (255, 255, 255), self.pos, 3)

    def draw_AABB(self, surf):
        pg.draw.line(surf, (255, 0, 0), self.AABB[:2], (self.AABB[2], self.AABB[1]), 2)
        pg.draw.line(surf, (255, 0, 0), (self.AABB[2], self.AABB[1]), (self.AABB[2:]), 2)
        pg.draw.line(surf, (255, 0, 0), self.AABB[2:], (self.AABB[0], self.AABB[3]), 2)
        pg.draw.line(surf, (255, 0, 0), (self.AABB[0], self.AABB[3]), self.AABB[:2])

    def update(self, dt):

        self.calculate_torque()
        self.calculate_angular_impulse()

        self.acc = scalar_mul(vec_add([F.components for F in self.forces]), self.inv_M) if self.forces else [0, 0]
        self.alpha = sum(self.torques) * self.inv_I
        
        delta_v = scalar_mul(self.acc, dt)
        self.vel = vec_add([self.vel, delta_v, vec_add([scalar_mul(v.components, self.inv_M) for v in self.impulses]) if self.impulses else [0, 0]])
        self.vel = scalar_mul(self.vel, self.damping)
        
        delta_p = scalar_mul(self.vel, dt)
        self.pos = vec_add([self.pos, delta_p])

        self.omega = self.omega + (self.alpha * dt) + (sum(self.angular_impulses) * self.inv_I)
        self.omega *= self.damping
        
        self.theta = self.theta + (self.omega * dt)
        
        if isinstance(self.shape, PolygonShape):
            self.shape.points = [vec_add([rotate_vector(p, self.theta), self.pos]) for p in self.shape.local_points]
            self.shape.edges = get_edges(self.shape.points)
            self.normals = get_normals(self.shape.edges)
            self.AABB = getAABB(self.shape.points)
        
        elif isinstance(self.shape, CircleShape):
            r = self.shape.radius
            x,y = self.pos
            self.AABB = [x-r, y-r, x+r, y+r]

    def add_force(self, forces):
        for i in forces:
            self.forces.append(i)
    
    def add_impulse(self, impulses):
        for i in impulses:
            self.impulses.append(i)

    def calculate_torque(self):
        for force in self.forces:
            rel_pos = vec_sub(force.pos, self.pos)
            torque = cross(rel_pos, force.components)
            self.torques.append(torque)

    def calculate_angular_impulse(self):
        for impulse in self.impulses:
            rel_pos = vec_sub(impulse.pos, self.pos)
            J = cross(rel_pos, impulse.components)
            self.angular_impulses.append(J)
            

    def clear_forces(self):
        self.forces = []
        self.torques = []
        self.impulses = []
        self.angular_impulses = []


    def collision_check(self, other):
        colliding = False
        overlap_x = self.AABB[0] <= other.AABB[2] and self.AABB[2] >= other.AABB[0]
    
        overlap_y = self.AABB[1] <= other.AABB[3] and self.AABB[3] >= other.AABB[1]
    
        if overlap_x and overlap_y: colliding = True

        if colliding:
            if isinstance(self.shape, PolygonShape) and isinstance(other.shape, CircleShape):
                max_d = -float("inf")
                best_i = -1

                for i in range(len(self.shape.local_points)):
                    c_v = vec_sub(other.pos, self.shape.points[i])
                    d = dot(c_v, self.normals[i])
                    if d > max_d:
                        max_d = d
                        best_i = i

                if max_d > other.shape.radius:
                    return None
                
                v1 = self.shape.points[best_i]
                v2 = self.shape.points[(best_i + 1) % len(self.shape.points)]
                e = vec_sub(v2, v1)
                c_v = vec_sub(other.pos, v1)

                t = dot(c_v, e)/dot(e, e)

                if 0 <= t <= 1:
                    best_normal = self.normals[best_i]
                    best_pen = other.shape.radius - max_d
                elif t < 0:
                    dist = math.sqrt(dot(c_v, c_v))
                    if dist > other.shape.radius:
                        return None
                    best_normal = scalar_mul(c_v, 1/dist) if dist != 0 else self.normals[best_i]
                    best_pen = other.shape.radius - dist
                elif t > 1:
                    c_v2 = vec_sub(other.pos, v2)
                    dist = math.sqrt(dot(c_v2, c_v2))
                    if dist > other.shape.radius:
                        return None 
                    best_normal = scalar_mul(c_v2, 1/dist) if dist != 0 else self.normals[best_i]
                    best_pen = other.shape.radius - dist

                contact = vec_sub(other.pos, scalar_mul(best_normal, other.shape.radius))

                return best_normal, best_pen, [contact]
            
            elif isinstance(self.shape, PolygonShape) and isinstance(other.shape, PolygonShape):
                N_a = self.normals
                N_b = other.normals

                axes = N_a + N_b


                min_overlap = float("inf")
                collision_normal = None

                for axis in axes:
                    max_a =  max_b = -float("inf")
                    min_b =  min_a = float("inf")

                    for v in self.shape.points:
                        p = dot(v, axis)
                        if p < min_a: 
                            min_a = p  
                        if p > max_a: 
                            max_a = p

                    for v in other.shape.points:
                        p = dot(v, axis)
                        if p < min_b: 
                            min_b = p
                        if p > max_b: 
                            max_b = p

                    if max_a < min_b or max_b < min_a:
                        return None
                    
                    overlap = min(max_a, max_b) - max(min_a, min_b)

                    if overlap < min_overlap:
                        min_overlap = overlap
                        collision_normal = axis

                    dx = other.pos[0] - self.pos[0]
                    dy = other.pos[1] - self.pos[1]

                    dot_prod = dot([dx, dy], collision_normal)

                    if dot_prod < 0:
                        collision_normal = scalar_mul(collision_normal, -1)

                best_a = -float("inf")
                best_b = float("inf")
                reference_face = None
                incident_face = None

                for i in range(len(self.shape.edges)):
                    d = dot(self.normals[i], collision_normal)
                    if d > best_a:
                        best_a = d
                        reference_face = self.shape.edges[i]
                        reference_index = i

                for i in range(len(other.shape.edges)):
                    d = dot(other.normals[i], collision_normal)
                    if d < best_b:
                        best_b = d
                        incident_face = other.shape.edges[i]

                e = vec_sub(reference_face[1], reference_face[0])
                t = scalar_mul(e, 1/math.sqrt(dot(e, e)))

                v1 = reference_face[0]
                v2 = reference_face[1]

                

                min_tx = dot(t, reference_face[0])
                max_tx = dot(t, reference_face[1])

                i1 = incident_face[0]
                i2 = incident_face[1]

                incident = [i1, i2]


                incident = clip(incident, scalar_mul(t, -1), -min_tx)

                incident = clip(incident, t, max_tx)

                if len(incident) == 0:
                    return None
                
                ref_normal = self.normals[reference_index]
                ref_c = dot(ref_normal, v1)

                contacts = []

                for p in incident:
                    separation = dot(ref_normal, p) - ref_c

                    if separation <= 0:
                        contacts.append(p)

                return collision_normal, min_overlap, contacts
         
                
    def resolve_collision(self, other, normal, pen, contacts):
        if self.static and other.static:
            return None
        for contact in contacts:
            total_inv = self.inv_M + other.inv_M

            correction = scalar_mul(normal, pen / total_inv)

            r_a = vec_sub(contact, self.pos)
            r_b = vec_sub(contact, other.pos)

            v_a = vec_add([self.vel, vector_cross(self.omega, r_a)])
            v_b = vec_add([other.vel, vector_cross(other.omega, r_b)])

            v_rel = vec_sub(v_b, v_a)

            vn = dot(v_rel, normal)

            if vn > 0:
                continue 

            num = -(1 + coeff_of_restitution) * vn

            ta = cross(r_a, normal)**2 * self.inv_I
            tb = cross(r_b, normal)**2 * other.inv_I

            denom = self.inv_M + other.inv_M + ta + tb

            j = num/denom

            j /= len(contacts)

            self.add_impulse([Vec2(contact, scalar_mul(normal, -j))])
            other.add_impulse([Vec2(contact, scalar_mul(normal, j))])

        self.pos = vec_sub(self.pos, scalar_mul(correction, self.inv_M))
        other.pos = vec_add([other.pos, scalar_mul(correction, other.inv_M)])

    
    def solve_two_body_constraint(self, other, J, bias):
        J_t = np.transpose(J)
        M_inv = np.diag([self.inv_M, self.inv_M, self.inv_I, other.inv_M, other.inv_M, other.inv_I])
        v_old = np.array([[self.vel[0]], [self.vel[1]], [self.omega], [other.vel[0]], [other.vel[1]], [other.omega]])
        num = -(J @ v_old) - bias
        denom = ((J @ M_inv) @ J_t)

        l = np.linalg.solve(denom, num)

        P = (M_inv @ J_t) @ l

        self.vel = vec_add([self.vel, P[:2, 0].tolist()])
        self.omega += P[2, 0]

        other.vel = vec_add([other.vel, P[3:5, 0].tolist()])
        other.omega += P[5, 0]

    def solve_one_body_constraint(self, J, bias):
        J_t = np.transpose(J)
        M_inv = np.diag([self.inv_M, self.inv_M, self.inv_I])
        v_old = np.array([[self.vel[0]], [self.vel[1]], [self.omega]])

        num = -(J @ v_old) - bias
        denom = ((J @ M_inv) @ J_t)

        l = np.linalg.solve(denom, num)

        P = (M_inv @ J_t) @ l 
        
        self.vel = vec_add([self.vel, P[:2, 0].tolist()])
        self.omega += P[2, 0]


    def position_constraint(self, other, local_pos_a, local_pos_b, L):
        rel_pos = vec_sub(other.pos, self.pos)
        current_dist = math.sqrt(dot(rel_pos, rel_pos))
        C = current_dist - L
        bias = (0.2/dt) * C
        n = scalar_mul(rel_pos, 1/math.sqrt(dot(rel_pos, rel_pos)))
        ra_n = -cross(local_pos_a, n)
        rb_n = cross(local_pos_b, n)
        return (np.array([[-n[0], -n[1], ra_n, n[0], n[1], rb_n]]), bias)
    
    
    def one_body_prismatic(self, local_pos, dt, x=None, y=None):
        r = rotate_vector(local_pos, self.theta)
        p = vec_add([self.pos, r])
        if x is not None:
            C = p[0] - x
            bias = (0.2/dt) * C
            return (np.array([[1, 0, -local_pos[1]]]), bias)
        elif y is not None:
            C = p[1] - y
            bias = (0.2/dt) * C
            return (np.array([[0, 1, local_pos[0]]]), bias)
        

    def one_body_revolute(self, pos, local_pos):
        r = rotate_vector(local_pos, self.theta)
        p = vec_add([self.pos, r])
        bias = np.array([[0.2/dt * (p[0] - pos[0])], [0.2/dt * (p[1] - pos[1])]])
        return (np.array([[1, 0, -r[1]], [0, 1, r[0]]]), bias)
    
    def two_body_revolute(self, other, local_pos_a, local_pos_b):
        r_a = rotate_vector(local_pos_a, self.theta)
        r_b = rotate_vector(local_pos_b, other.theta)

        p_a = vec_add([self.pos, r_a])
        p_b = vec_add([other.pos, r_b])

        bias = np.array([[0.2/dt * (p_b[0] - p_a[0])], [0.2/dt * (p_b[1] - p_a[1])]])
        return (np.array([[-1, 0, r_a[1], 1, 0, -r_b[1]],
                          [0, -1, -r_a[0], 0, 1, r_b[0]]]), bias)


#part of RL
class state:
    def __init__(self, pos, vel, theta, omega):
        self.pos = box.pos[0]
        self.vel = box.vel[0]
        self.theta = rod.theta
        self.omega = rod.omega

#walls
wall_r = RigidBody(shape=PolygonShape(points=[(50, -HEIGHT/2), (-50, -HEIGHT/2), (-50, HEIGHT/2), (50, HEIGHT/2)], color=(0, 0, 0, 0)), 
                   M=None, I=None, pos=[WIDTH + 50, HEIGHT/2], static=True)

wall_l = RigidBody(shape=PolygonShape(points=[(50, -HEIGHT/2), (-50, -HEIGHT/2), (-50, HEIGHT/2), (50, HEIGHT/2)], color=(0, 0, 0, 0)), 
                   M=None, I=None, pos=[-50, HEIGHT/2], static=True)

wall_u = RigidBody(shape=PolygonShape(points=[(WIDTH/2, -50), (-WIDTH/2, -50), (-WIDTH/2, 50), (WIDTH/2, 50)], color=(0, 0, 0, 0)), 
                   M=None, I=None, pos=[WIDTH/2, -50], static=True)

wall_d = RigidBody(shape=PolygonShape(points=[(WIDTH/2, -50), (-WIDTH/2, -50), (-WIDTH/2, 50), (WIDTH/2, 50)], color=(0, 0, 0, 0)), 
                   M=None, I=None, pos=[WIDTH/2, HEIGHT + 50], static=True)



#box and rod
rod_shape = PolygonShape(points=[(4, -50), (-4, -50), (-4, 50), (4, 50)], color=(255, 165, 0))
box_shape = PolygonShape(points=[(20, -10), (-20, -10), (-20, 10), (20, 10)], color=(0, 0, 0))

box = RigidBody(shape=box_shape, M=1, I=1/6  * 1 * 10**2, pos=[WIDTH/2, HEIGHT/2], rotatable=False)
rod = RigidBody(shape=rod_shape, M=0.1, I=1/12 * 0.1 * 100**2, pos=[WIDTH/2, HEIGHT/2 - 50], theta=180)

all_objects = [box, rod, wall_l, wall_r, wall_u, wall_d]
objects = [box, rod]
walls = [wall_l, wall_r, wall_u, wall_d]


pg.init()


pg.font.init() 
my_font = pg.font.SysFont("Arial", 24, bold=True)

screen = pg.display.set_mode((WIDTH, HEIGHT))

fps = 100
dt = 1/fps

clock = pg.time.Clock()

run = True

debug_contacts = []
DEBUG_LIFETIME = 100 

while run:
    for obj in all_objects:
        obj.clear_forces()

    
    J = Vec2(box.pos, [10, 0])
    J2 = Vec2(box.pos, [-10, 0])

    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

        #tap key to apply impulse
        """ if event.type == pg.KEYDOWN:
            if event.key == pg.K_d:
                 box.add_impulse([J])
            elif event.key == pg.K_a:
                box.add_impulse([J2]) """


    state_before_action = state
    for obj in objects:
        obj.add_force([Vec2(obj.pos, [0, obj.M * 981])])

    
    #hold key to apply impulse
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        box.add_impulse([J])
    
    if keys[pg.K_a]:
        box.add_impulse([J2])
        
    
    box.solve_one_body_constraint(*box.one_body_prismatic(local_pos=[0, 0], dt=dt, y=HEIGHT/2))
    rod.solve_two_body_constraint(box, *rod.two_body_revolute(box, [0, -50], [0, 0]))
    state_after_action = state
    for i in range(len(walls)):
        for j in range(len(objects)):
            npc = walls[i].collision_check(objects[j])
            if npc:
                n, p, c = npc
                walls[i].resolve_collision(objects[j], n, p, c)
                for k in c:
                    debug_contacts.append([k, DEBUG_LIFETIME])
          
    for object in all_objects:
        object.update(dt)
    

    screen.fill((100,100, 100))

    pg.draw.line(screen, (0, 0, 0), (0, HEIGHT/2), (WIDTH, HEIGHT/2), 2)

    for i in range(len(debug_contacts) - 1, -1, -1):
        point = debug_contacts[i][0]
        life = debug_contacts[i][1]
        
        
        draw_x, draw_y = int(point[0]), int(point[1])
        
       
        color_val = int((life / DEBUG_LIFETIME) * 255)
        color = (color_val, color_val, color_val)
        
        pg.draw.circle(screen, color, (draw_x, draw_y), 4)
        
     
        debug_contacts[i][1] -= 1
    
        if debug_contacts[i][1] <= 0:
            debug_contacts.pop(i)

    for object in all_objects:
        object.draw_body(screen, object.shape.color)

    
    
    current_fps = str(int(clock.get_fps()))
    
    
    fps_surface = my_font.render(f"FPS: {current_fps}", True, (255, 255, 0)) 

    screen.blit(fps_surface, (10, 10))

    pg.display.flip()
    clock.tick(fps)
    
    
pg.quit()



