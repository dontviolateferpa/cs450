from random import Random
import numpy

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
    movie_list = []

    for x in range (0, 5):
        movie_list.append(movie("blah " + str(x), 2000 + x, x * 60))

    return movie_list

def create_list_of_long_movies(movies):
    """"""
    long_movie_list = []

    for movie in movies:
        if movie.runtime > 150:
            long_movie_list.append(movie)
    
    return long_movie_list

def main():
    """"""
    # create list of movies
    movie_list = create_movie_list()

    # print movie list
    for movie in movie_list:
        print(movie)

    print()

    # make list of long movies
    long_movie_list = create_list_of_long_movies(movie_list)

    for movie in long_movie_list:
        print(movie)
    
    movies_stars_map = {}

    random = Random()

    for movie in movie_list:
        num_stars = random.uniform(0, 5)

        movies_stars_map[movie.title] = num_stars
    
    for movie_title in movies_stars_map:
        num_stars = movies_stars_map[movie_title]
        
        print "{0:.2f}".format(num_stars)

main()
