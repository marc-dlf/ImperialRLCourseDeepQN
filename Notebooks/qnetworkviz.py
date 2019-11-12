import cv2
import numpy as np

class QNetworkViz(object):

    def __init__(self):
        self.height = 500
        self.width = 500
        self.square_size = self.height/10
        self.img = np.zeros([self.height,
                             self.width,
                             3], dtype=np.uint8)

    def get_color(self,qvalue):
        val = int(qvalue*255)
        return [255-val,val,val]


    def normalize_qvalues(self,q):
        normalized_q = np.copy(q)
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                normalized_q[i,j,:] = (normalized_q[i,j,:] - np.min(normalized_q[i,j,:]))/np.ptp(normalized_q[i,j,:])

        return normalized_q

    def plot_q_values(self,q):

        normalized_q = self.normalize_qvalues(q)
        normalized_q = np.flip(normalized_q,axis=1)

        for i in range(0,10):
            for j in range(0,10):
            ### Points coordinates

                top_left_point = (int(i*(self.square_size))-(i>0),
                                  int(j*(self.square_size))-(j>0))


                bottom_left_point = (int(i*(self.square_size))-(i>0),
                                     int((j+1)*(self.square_size))-1)

                top_right_point = (int((i+1)*(self.square_size))-1,
                                  int(j*(self.square_size))-(j>0))

                bottom_right_point = (int((i+1)*(self.square_size))-1,
                                      int((j+1)*(self.square_size))-1)

                middle_point = (int(self.square_size*(i+(1/2)))-1,
                                int(self.square_size*(j+(1/2)))-1)

                ### Right triangles

                right_triangle = [[top_right_point,bottom_right_point],
                                  [bottom_right_point,middle_point],
                                  [middle_point,top_right_point]]

                cv2.fillConvexPoly(self.img,np.array(right_triangle,dtype=np.int32).reshape(-1,1,2)
                                   ,color=self.get_color(normalized_q[i,j,0]),lineType=8)

                cv2.line(self.img,right_triangle[1][0],right_triangle[1][1],color=(0,0,0))
                cv2.line(self.img,right_triangle[2][0],right_triangle[2][1],color=(0,0,0))

                ### Down triangles

                down_triangle = [[top_left_point,top_right_point],
                                 [top_right_point,middle_point],
                                 [middle_point,top_left_point]]


                cv2.fillConvexPoly(self.img,np.array(down_triangle,dtype=np.int32).reshape(-1,1,2)
                                   ,color=self.get_color(normalized_q[i,j,3]))

                cv2.line(self.img,down_triangle[1][0],down_triangle[1][1],color=(0,0,0))
                cv2.line(self.img,down_triangle[2][0],down_triangle[2][1],color=(0,0,0))

                ### Left triangles

                left_triangle = [[top_left_point,bottom_left_point],
                                 [bottom_left_point,middle_point],
                                 [middle_point,top_left_point]]


                cv2.fillConvexPoly(self.img,np.array(left_triangle,dtype=np.int32).reshape(-1,1,2),
                                   color=self.get_color(normalized_q[i,j,2]))

                cv2.line(self.img,left_triangle[1][0],left_triangle[1][1],color=(0,0,0))
                cv2.line(self.img,left_triangle[2][0],left_triangle[2][1],color=(0,0,0))

                ### Up triangles
                up_triangle = [[bottom_right_point,bottom_left_point],
                               [bottom_left_point,middle_point],
                               [middle_point,bottom_right_point]]

                cv2.fillConvexPoly(self.img,np.array(up_triangle).reshape(-1,1,2),
                                   color=self.get_color(normalized_q[i,j,1]))

                cv2.line(self.img,up_triangle[1][0],up_triangle[1][1],color=(0,0,0))
                cv2.line(self.img,up_triangle[2][0],up_triangle[2][1],color=(0,0,0))

    def plot_grid(self):
        for i in range(1,10):
            cv2.line(self.img,(0,int(i*(self.height/10)-1)),
                     (self.width-1,int(i*(self.height/10))-1),
                     (255,255,255))
            cv2.line(self.img,(int(i*(self.width/10))-1,0),
                     (int(i*(self.width/10))-1,self.height-1),
                     (255,255,255))


    def get_pixel(self,state):
        pixel_x = int(self.width*(state[0]))
        pixel_y = int(self.height - self.height*(state[1]))

        return (pixel_x,pixel_y)

    def plot_path(self,successive_states):
        first_state_pixel = self.get_pixel(successive_states[0])
        last_state_pixel = self.get_pixel(successive_states[-1])


        delta = 255/(len(successive_states)-1)
        for i in range(len(successive_states)-1):

            current_pixel = self.get_pixel(successive_states[i])
            next_pixel = self.get_pixel(successive_states[i+1])
            cv2.line(self.img,current_pixel,
                         next_pixel,
                         (0,int(i*delta),255-int(i*delta)),thickness=3)

        cv2.circle(self.img, first_state_pixel, 5, (0,0,255), cv2.FILLED)
        cv2.circle(self.img, last_state_pixel, 5, (0,255,0), cv2.FILLED)


    def show(self):
        cv2.imshow("Environment", self.img)
        cv2.waitKey(1)

    def reset(self):
        self.img = np.zeros([self.height,
                             self.width,
                             3], dtype=np.uint8)
