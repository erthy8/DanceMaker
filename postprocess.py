import numpy as np

def getAngles(series):
    result = []
    result.append(calculate_angle(series[12], series[14], series[16])) #left elbow - 12, 14, 16
    result.append(calculate_angle(series[11], series[13], series[15])) #right elbow - 11,13,15
    result.append(calculate_angle(series[14], series[12], series[24])) #left armpit - 14,12,24
    result.append(calculate_angle(series[13], series[11], series[23])) #right armpit - 13,11,23
    result.append(calculate_angle(series[12], series[24], series[26])) #left hip - 12,24,26
    result.append(calculate_angle(series[11], series[23], series[25])) #right hip - 11,23,25
    result.append(calculate_angle(series[24], series[26], series[28])) #left knee - 24,26,28
    result.append(calculate_angle(series[23], series[25], series[27])) #right knee - 23,25,27
    return result


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

if name == "__main__": 
    df1 = ... #path to uploaded dance dataframe
    df2 = ... #path to live dance dataframe
    score = 0
    live_angles = []
    vid_angles = []
    #elbow angle, hip angle, knee angle, armpit angle
    for i in range(min(len(df1, df2))):
        live_angles.append(getAngles(df1[i]))
        vid_angles.append(getAngles(df2[i]))
    #linear regression     
    x = 
        
        
    

