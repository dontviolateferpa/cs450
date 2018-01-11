class movie:
    title = str()
    year = int()
    runtime = int()

    def __init__(self, title, year, runtime):
        self.title = title
        self.year = year
        if (runtime < 0):
            self.runtime = 0
        else:
            self.runtime = runtime

    def __repr__(self):
        return ("" + self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins")

    def runtime_mins_hours(self):
        """return runtime in mins and hours"""
        return self.runtime, self.runtime / 60.0

def create_movie_list():
    """"""
    movieList = []

    for x in range (0, 4):
        movieList.append(movie("blah " + str(x), 2000 + x, x))

    return movieList

def main():
    """"""""
    thisMovieList = create_movie_list()

    for x in thisMovieList:
        print(x)

main()