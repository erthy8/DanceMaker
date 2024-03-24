import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def getAngles(series):
    result = []
    # print(series[12].landmark[1])
    # lndmrk = series[12].landmark[1]
    result.append(calculate_angle(
        series[12], series[14], series[16]))  # left elbow
    result.append(calculate_angle(
        series[11], series[13], series[15]))  # right elbow
    result.append(calculate_angle(
        series[14], series[12], series[24]))  # left armpit
    result.append(calculate_angle(
        series[13], series[11], series[23]))  # right armpit
    result.append(calculate_angle(
        series[12], series[24], series[26]))  # left hip
    result.append(calculate_angle(
        series[11], series[23], series[25]))  # right hip
    result.append(calculate_angle(
        series[24], series[26], series[28]))  # left knee
    result.append(calculate_angle(
        series[23], series[25], series[27]))  # right knee
    return result


def calculate_angle(a, b, c):
    # a = np.array(a)  # First
    # b = np.array(b)  # Mid
    # c = np.array(c)  # End

    radians = np.arctan2(c.y-b.y, c.x-b.x) - \
        np.arctan2(a.y-b.y, a.x-b.x)
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def postProcessScore():
    title = ""
    with open('logs.txt', 'r') as file:
        # Iterate over lines
        for line in file:
            title = line
            break  # Break after the first iteration

    # path to uploaded dance dataframe
    df1 = pd.read_hdf(f'data/{title}/livedata.hdf5', key="livedata")
    # path to live dance dataframe
    df2 = pd.read_hdf(f'data/{title}/videodata.hdf5', key="videodata")
    live_angles = []
    vid_angles = []
    # elbow angle, hip angle, knee angle, armpit angle
    for i in range(min(len(df1), len(df2))):
        if df1.iloc[i, 0] != None and df2.iloc[i, 0] != None:
            live_angles.append(getAngles(df1.iloc[i, 0:]))
            vid_angles.append(getAngles(df2.iloc[i, 0:]))

    # linear regression
    x = np.array(vid_angles)
    y = np.array(live_angles)

    final_array = (1 - (np.abs(x - y) / x)) * 100
    average = np.mean(final_array)

    model = LinearRegression().fit(x, y)
    score = model.coef_
    score = model.score(x, y)
    return score

    print(f"average: {average}")
    print(f"linear regression: {model.score(x,y)}")
