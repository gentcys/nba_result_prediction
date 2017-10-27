import pandas
import math
import csv
import random
import numpy
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


# 当每支队伍没有elo等级分时，赋予其基础elo等级分
base_elo = 1600
team_elos = {}
team_stats = {}
x = []
y = []
folder = 'data'

# 根据每支队伍的Micellaneous, Opponent, Team统计数据csv文件进行初始化
def initialize_data(miscellaneous_stats, opponent_per_game_stats, team_per_game_stats):
    miscellaneous_stats.drop(['Rk', 'Arena'], axis=1, inplace=True)
    opponent_per_game_stats.drop(['Rk', 'G', 'MP'], axis=1, inplace=True)
    team_per_game_stats.drop(['Rk', 'G', 'MP'], axis=1, inplace=True)

    team_stats = pandas.merge(miscellaneous_stats, opponent_per_game_stats, how='left', on='Team')
    team_stats = pandas.merge(team_stats, team_per_game_stats, how='left', on='Team')
    return team_stats.set_index('Team', inplace=False, drop=True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]

# 计算每个球队的elo值
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))

    #根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16

    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank

def build_season_data(all_data):
    print("Building season data.")
    X = []
    skip = 0

    for index, row in all_data.iterrows():

        # Get starter or previous elos.
        Wteam = row['WTeam']
        Lteam = row['LTeam']

        # 获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        # 给主场比赛的队伍加上100的elo值
        if row['WVenue'] == 'Home':
            team1_elo += 100
        else:
            team2_elo += 100


        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加我们从basketball-reference.com获得的每个队伍的统计信息
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)

        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        # 讲两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        if skip == 0:
            print(X)
            skip = 1

        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return numpy.nan_to_num(X), y

def predict_winner(team_1, team_2, model):
    features = []

    # team 1, Away team
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # team 2, Home team
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)

    features = numpy.nan_to_num(features)
    return model.predict_proba([features])


if __name__ == '__main__':
    miscellaneous_stats = pandas.read_csv('data/MiscellaneousStats.csv')
    opponent_per_game_stats = pandas.read_csv('data/OpponentPerGameStats.csv')
    team_per_game_stats = pandas.read_csv('data/TeamPerGameStats.csv')

    team_stats = initialize_data(miscellaneous_stats, opponent_per_game_stats, team_per_game_stats)

    result_data = pandas.read_csv('data/2016-2017-ScheduleAndResults-processed.csv')
    X, y = build_season_data(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())

    #利用训练好的model在17-18年的比赛中进行预测
    print("Predicting on new schedule")
    schedules = pandas.read_csv('data/2017-2018-Schedules-processed.csv')
    result = []

    for index, row in schedules.iterrows():
        date = row['Date']
        team1 = row['Away']
        team2 = row['Home']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([date, winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([date, winner, loser, 1 - prob])

    with open('data/2017-2018-Results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'win', 'lose', 'probability'])
        writer.writerows(result)
