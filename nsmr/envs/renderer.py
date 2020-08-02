from nsmr.envs.consts import *

class Renderer(object):
    def __init__(self, dimentions):
        self.viewer = None
        self.margin = 0.2
        screen_size = 600
        self.SKIP_RENDER = 20
        world_width_x = dimentions[0]*RESOLUTION + self.margin * 2.0
        world_width_y = dimentions[1]*RESOLUTION + self.margin * 2.0
        self.scale = screen_size / max(world_width_x, world_width_y)
        self.screen_width = int(world_width_x*self.scale)
        self.screen_height = int(world_width_y*self.scale)

    def render(self, state, mode):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            #wall
            l, r, t, b = self.get_lrtb(0,state.dimentions[0],0,state.dimentions[1])
            wall = rendering.PolyLine([(l,b),(l,t),(r,t),(r,b)],True)
            wall.set_color(0.,0.,0.)
            self.viewer.add_geom(wall)
            #robot
            robot = rendering.make_circle(ROBOT_RADIUS*self.scale)
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            robot.set_color(0.0,0.0,1.0)
            self.viewer.add_geom(robot)
            robot_orientation = rendering.make_capsule(ROBOT_RADIUS*self.scale,1.0)
            self.orientation_trans = rendering.Transform()
            robot_orientation.set_color(0.0,1.0,0.0)
            robot_orientation.add_attr(self.orientation_trans)
            self.viewer.add_geom(robot_orientation)
            #target
            target = rendering.make_circle(ROBOT_RADIUS*0.3*self.scale)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target.set_color(1.0,0.0,0.0)
            self.viewer.add_geom(target)
            #obstract
            for obj in state.layout["static_objects"]:
                l, r, t, b = self.get_lrtb(obj["l"],obj["r"],obj["t"],obj["b"])
                ob = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
                ob.set_color(0,0,0)
                self.viewer.add_geom(ob)

        robot_x = (self.margin + state.pose[0]) * self.scale
        robot_y = (self.margin + state.pose[1]) * self.scale
        robot_orientation = state.pose[2]
        self.robot_trans.set_translation(robot_x, robot_y)
        self.orientation_trans.set_translation(robot_x,robot_y)
        self.orientation_trans.set_rotation(robot_orientation)
        self.target_trans.set_translation((state.target[0]+self.margin)*self.scale,(state.target[1]+self.margin)*self.scale)
        if state.obs is not None:
            for i in range(int(len(state.obs))):
                if i%self.SKIP_RENDER == 0:
                    lidar = rendering.make_capsule(self.scale*state.obs[i],1.0)
                    lidar_trans = rendering.Transform()
                    lidar_trans.set_translation(robot_x,robot_y)
                    lidar_trans.set_rotation(state.pose[2] + i*ANGLE_INCREMENT - MAX_ANGLE)
                    lidar.set_color(1.0,0.0,0.0)
                    lidar.add_attr(lidar_trans)
                    self.viewer.add_onetime(lidar)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_lrtb(self, l, r, t, b):
        l = (self.margin+l*RESOLUTION) * self.scale
        r = (self.margin+r*RESOLUTION) * self.scale
        t = (self.margin+t*RESOLUTION) * self.scale
        b = (self.margin+b*RESOLUTION) * self.scale
        return l, r, t, b

    def close(self):
        if self.viewer:
            self.viewer.close()
