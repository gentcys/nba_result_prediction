import sys
import pandas


class ScheduleResultsProcesser(object):
    '''
    Read schedule result file, deal with its contents and return a new file within the content formatted like this:
    WTeam,LTeam,WVenue
    New Your Knicks,Brooklyn Nets,Vistor
    Toronto Raptors,Boston Celtics,Home
    Minnesota Timberwolves,Cleveland Cavaliers,Home
    ...
    '''

    def main(self, schedule_results, filename):
        schedule_results.drop(schedule_results.columns[[0,1,6,7,8]], axis=1, inplace=True)
        schedule_results = schedule_results.assign(WVenue = '')

        for i, row in schedule_results.iterrows():
            row['WVenue'] = 'Away'

            if row['PTS'] < row['PTS.1']:
                row['Visitor/Neutral'], row['Home/Neutral'] = row['Home/Neutral'], row['Visitor/Neutral']
                row['WVenue'] = 'Home'

            schedule_results.iloc[i] = row

        schedule_results.drop(schedule_results.columns[[1,3]], axis=1, inplace=True)

        schedule_results.columns = ['WTeam', 'LTeam', 'WVenue']

        filename = filename[:-4] + '-processed' + filename[-4:]

        schedule_results.to_csv(filename, index=False)


if __name__ == "__main__":
    filename = sys.argv[1]
    schedule_results = pandas.read_csv(filename)
    processer = ScheduleResultsProcesser()
    results = processer.main(schedule_results,filename)
