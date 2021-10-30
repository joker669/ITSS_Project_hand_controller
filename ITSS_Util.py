import math

def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    #if included_angle > 180:
      #included_angle = 360 - included_angle
  return included_angle
  
  
  
def reg_five(lmList1):
    AB = [lmList1[8][0],lmList1[8][1],lmList1[0][0],lmList1[0][1]]
    CD = [1,0,0,0]
    ang = angle(AB, CD)
    if(ang >45 and ang <= 135):
        return 3
    elif(ang > 135 and ang <= 225):
        return 2
    elif(ang > 225 and ang <= 315):
        return 4
    else:
        return 1
    