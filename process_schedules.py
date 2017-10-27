import sys
import pandas


class SchedulesProcesser(object):
    '''
    Read schedules file, deal with its contents and return a new file within the content formatted like this:
    Away,Home
    New Your Knicks,Brooklyn Nets
    Toronto Raptors,Boston Celtics
    Minnesota Timberwolves,Cleveland Cavaliers
    ...
    '''

    def main(self, schedules, filename):
        schedules.drop(schedules.columns[[0,1,3,5,6,7,8]], axis=1, inplace=True)

        schedules.columns = ['Away', 'Home']

        filename = filename[:-4] + '-processed' + filename[-4:]

        schedules.to_csv(filename, index=False)


if __name__ == "__main__":
    filename = sys.argv[1]
    schedules = pandas.read_csv(filename)
    processer = SchedulesProcesser()
    results = processer.main(schedules,filename)
