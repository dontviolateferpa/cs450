"""
Author: Scott Burton
Purpose: This program helps practice the syntax of Python. It
    defines a Movie class and then performs various operations on it.
"""

# First we import the libraries that we'll use
from random import Random
import numpy as np

# Python doesn't have true constants, we just make variables
# with all caps, so we remember not to change them.
MINUTES_IN_HOUR = 60


class Movie:
    """
    Represents a movie (e.g., Netflix movie)
    """
    def __init__(self, title="", year=0, runtime=0):
        """
        Constructs a new movie. Notice the use of default params
        rather than trying to make an overloaded non-default constructor
        :param title: The new title
        :param runtime: The runtime in minutes
        """
        self.title = title
        self.year = year

        if runtime > 0:
            self.runtime = runtime
        else:
            self.runtime = 0

    def get_runtime_hours_and_minutes(self):
        """
        Returns the runtime in hours and minutes.
        :return: hours, minutes
        """
        # In Python 3, you need to use the // to do integer division
        hours = self.runtime // MINUTES_IN_HOUR
        minutes = self.runtime % MINUTES_IN_HOUR

        # Yes, you can return two things at once!
        return hours, minutes

    def __repr__(self):
        """
        Returns a string representation of the movie
        """
        return "{} ({}) - {} mins".format(self.title, self.year, self.runtime)


def create_movie_list():
    """
    Returns a list of movies.
    :return:
    """

    # This creates a new list (they work like Java ArrayLists or C++ Vectors)
    movies = []

    # There are lots of ways to create these movies, here we will demo a few
    m1 = Movie("Star Wars: Episode VII - The Force Awakens", 2015, 135)
    movies.append(m1)

    m2 = Movie()
    m2.title = "Avatar"
    m2.year = 2009
    m2.runtime = 162
    movies.append(m2)

    movies.append(Movie("Titanic", 1997, 195))
    movies.append(Movie(runtime=124, year=2015, title="Jurassic World"))

    return movies


def part2():
    """
    Runs through the steps of Part 2 of the assignment
    :return:
    """
    movies = create_movie_list()

    # Iterating through lists is super easy!
    print("All movies:")
    for movie in movies:
        print(movie)

    # Print a blank line
    print()

    # This list comprehension creates a new list by iterating through
    # our current one and only using those that match the criteria
    long_movies = [movie for movie in movies if movie.runtime > 150]

    # Loop through and display the long movies
    print("Long movies:")
    for movie in long_movies:
        print(movie)
    print()

    # We'll use this for random numbers in a minute...
    random = Random()

    # This creates a new Dictionary (i.e., HashMap or HashTable)
    stars_map = {}

    # Go through our movies and add them to the the dictionary
    for movie in movies:
        # Get a random number
        stars_value = random.uniform(0, 5)

        # Set the new key-value pair into the dictionary
        stars_map[movie.title] = stars_value

    # Go through each thing in the map
    for title in stars_map:
        # Get the value
        stars = stars_map[title]

        # Print it out, formatted appropriately
        print("{} - {:.2f} stars".format(title, stars))


def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, I just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array


def part3():
    """
    Runs through each step in Part 3
    :return:
    """
    data = get_movie_data()

    # this line keeps it from showing scientific notation
    np.set_printoptions(suppress=True)
    
    print(data)
    print(data.shape)

    rows = data.shape[0]
    print("Rows:", rows)

    cols = data.shape[1]
    print("Cols:", cols)

    print("\nFirst two rows:")
    first_two_rows = data[0:2]
    print(first_two_rows)

    print("\nLast two columns:")
    last_two_cols = data[:, -2:]
    print(last_two_cols)

    views = data[:, 1]
    print(views)


def main():
    m = Movie("Star Wars", 1977, 125)
    hours, mins = m.get_runtime_hours_and_minutes()
    print("Hours: {}, Minutes: {}".format(hours, mins))

    part2()
    part3()


# While not required, it is considered good practice to have
# a main function and use this syntax to call it.
if __name__ == "__main__":
    main()
