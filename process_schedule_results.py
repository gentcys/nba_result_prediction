class ScheduleResultsProcesser(object):
    '''
    Read schedule result files, deal with their contents and return a new file within the content formatted like this:
    WTeam,LTeam,WLoc
    New Your Knicks,Brooklyn Nets,Vistor
    Toronto Raptors,Boston Celtics,Home
    Minnesota Timberwolves,Cleveland Cavaliers,Home
    ...
    '''
