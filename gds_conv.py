import re
import numpy as np

class GDS_file:

    def __init__(self, file):
        #file = open(filename, 'r') # opens the file
        self.file_strings = file.readlines() # opens the file and returns it as a list of strings
        #file.close()
        self.shapes = self.to_shapes()
        self.num_shapes = len(self.shapes)

    def to_shapes(self):
        # takes in the list of strings and returns a dictionary of the shapes. Each shape is given by its own
        # dictionary that contains which layer it is in and its coordinates (as an array, with the 0th column being 
        # the x_coordinates and the 1st column being the y coordinates)
        indices = [i for i, x in enumerate(self.file_strings) if x == "BOUNDARY \n"] # tell me the indices of 'Boundary' in the list. 
        # The actual coordinates of the shape then start 3 items later and finish when we come to an 'ENDEL' item
        shapes = {}
        for i in indices:
            shape = indices.index(i)
            shapes[shape] = {}
            shapes[shape]['layer'] = int(self.file_strings[i+1][-3])
            n=i+3
            coords = []
            while self.file_strings[n] != 'ENDEL \n':
                x = float(re.search('-?[0-9]+', self.file_strings[n])[0])
                y = float(re.search('[:]\s-?[0-9]+', self.file_strings[n])[0][2:])
                coords.append([x,y])
                n += 1
                shapes[shape]['coordinates'] = np.array(coords.copy())
        return shapes

#example = GDS_file('4-contact-STM.txt')


# what's left? 