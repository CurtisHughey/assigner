#!/usr/bin/env python3

# pip3 install munkres pandas numpy openpyxl regex argparse matplotlib

from munkres import Munkres, print_matrix

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import argparse


def weight_function(x, a, b, maximum):
    return maximum / (1 + ((maximum/x)**a - 1) ** b)



def calc_midpoint(c, m):
    return 1 / (math.log2(m/c))


def graph_function(b, skew, maximum):
    axis_color = "lightgoldenrodyellow"

    fig = plt.figure()
    ax = fig.add_subplot(111)  # TODO what's the 111

    c = maximum * skew
    a = calc_midpoint(c, maximum)
    b = b

    fig.subplots_adjust(left=0.25, bottom=0.25)

    x = np.arange(0.0001, maximum, 0.01)
    [line] = ax.plot(x, weight_function(x, a, b, maximum))
    ax.set_xlim([0, maximum])
    ax.set_ylim([0, maximum])
    
    skew_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
    skew_slider = Slider(skew_slider_ax, "Skew", 0, 1, valinit=skew)
    
    steepness_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03], facecolor=axis_color)
    steepness_slider = Slider(steepness_slider_ax, "Steepness", 1, 10, valinit=b)
    
    def sliders_on_changed(val):
        line.set_ydata(weight_function(x, calc_midpoint(maximum*skew_slider.val, maximum), steepness_slider.val, maximum))
        fig.canvas.draw_idle()
    skew_slider.on_changed(sliders_on_changed)    
    steepness_slider.on_changed(sliders_on_changed)
    
    
    #plt.plot(x, y)
    plt.show()  # This just doesn't seem to work in Jupyter... ok
    
    return skew_slider.val, steepness_slider.val

def drop_names(df):
    names = df[df.columns[0]].values
    df = df.drop(df.columns[0], axis=1)
    return df

def weight_input(df, b, skew):
    maximum = df.max(axis=1)[0]  # Might want all the max's...

    a, b = graph_function(b, skew, maximum)

    print("================\na,b: {}, {}\n================".format(a, b))

    curried_function = lambda x: weight_function(x, a, b, maximum) 

    df = df.apply(curried_function, axis=1)
    
    return df
    
    
def resolve_column_names(df):
    original_name_dict = {}

    multiplier_fmt = r"\(x\d+\)"
    for c in df:
        print(c)
        m = re.search(multiplier_fmt, c)
        
        if m:
            number_fmt = r"\d+"
            number = int(re.search(number_fmt, c[m.start():m.end()])[0])
            core = c[:m.start()].strip()
            
            # Now we have the core name and the number of times the column should be replicated
            if number >= 1:
                for i in range(number):
                    new_column = "{} (#{})".format(core, i+1)
                    #df[new_column] = df[c]
                    df.insert(loc=df.columns.get_loc(c), column=new_column, value=df[c].tolist())
                    original_name_dict[new_column] = c  # Make sure to point back to the original when we reconstruct the matrix
                # Now delete original
                df = df.drop(c, axis=1)
            else:
                print("Improper multiplier ({}), continuing".format(number))

    return df, original_name_dict


def do_munkres(df):
    matrix = df.to_numpy()

    master_matrix = matrix.copy()

    # Use Munkres
    m = Munkres()
    print("=" * 40)
    indices = m.compute(matrix)

    print_matrix(matrix)

    print("=" * 40)

    total = 0
    for r, c in indices:
        total += master_matrix[r][c]
        print("({}, {}) -> {}".format(r, c, master_matrix[r][c]))

    print(indices)
    print(total)

    return indices

# Now for the cleared out matrix, alert if the student and site have matched
def prep_output(df, df_output, original_name_dict, indices):
    df_copy = df_output.copy()  # Need to save this off so I can remember the picks for each person when I output
    
    # Overwrite the existing matrix, initially clearing out    
    for column in df_output:
        if column.lower() == "student":
            continue
        else:
            df_output[column] = ""


    for r, c in indices:
        r = r
        c = c  # Adjusting for the row and column names  (TODO guess we don't need to do this anymore)

        # Now we look up in matrix for the corresponding row and column. Then we take that column and look up in the master_df (we had created column copies)
        column = df.columns[c]
        print(original_name_dict.get(column, "NONE"))

        actual_column_name = original_name_dict.get(column, column)  # Attempt a lookup, otherwise it was unchanged

        df_output.loc[r, actual_column_name] = "Choice: {}".format(df_copy[actual_column_name].iloc[r])

    return df_output


# Gets the table of students and sites in the format I like:
# Students on x axis
# Sites on y axis. Sites are short name, and have number of slots in parentheses
# Each intersection has the numeric ranking of the student for that site
def get_formatted_students_sites_table(definitions_filename, input_filename):

    data = {"Student": []}

    ####################################################################

    # Reads the definitions file (columns of name, long name (survey name), and number of slots), returns a tuple of dicts: name_lookup which allows you to look up the short name from the long name, and preferences_lookup, which allows you to look up the formatted column of slots we want for the next step
    df_definitions = pd.read_excel(definitions_filename)
    
    print(df_definitions)
    
    preferences_lookup = {}
    for name, slots in zip(df_definitions["Name"], df_definitions["Slots"]):
        name = name.strip()
        if int(slots) == 1:
            key = name
        else:
            key = "{} (x{})".format(name, int(slots))
        data[key] = []
        preferences_lookup[name] = key     
    
    name_lookup = {}
    for name, longname in zip(df_definitions["Name"], df_definitions["Survey Name"]):
        stripped_longname = re.sub(r"\s+", "", longname)
        name_lookup[stripped_longname] = name

    ######################################################################

    df = pd.read_excel(input_filename)

    # So sometimes there's a \xa0 character... NBSP... that's annoying. Different sort of space character
    for name, preferences in zip(df["Name"], df["Rank your site preferences below."]):
        name = name.strip()
        preferences = preferences
        preferences = [name_lookup.get(re.sub(r"\s+", "", x), None) for x in preferences.split(";")]
        
        if not preferences[-1]:  # Usually seems to be a trailing semicolon
            preferences = preferences[:-1]
        
        data["Student"].append(name)
        
        for idx, preference in enumerate(preferences):
            data[preferences_lookup[preference]].append(idx+1)  # idx is 0-based, convert to start at 1
        
    df = pd.DataFrame(data)
    
    return df


def main():
    import socket
    hostname = socket.gethostname()
    print(hostname)
    if "jupyter" in hostname:
        print("Running in a Jupyter context!!")
    else:
        print("Probably running in a command-line context")

    #############################

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input file")
    parser.add_argument("--definition-file", type=str, required=True, help="Definitions file")
    parser.add_argument("--output-file", type=str, required=True, help="Output file")
    parser.add_argument("--steepness", type=int, required=False, default=3, help="Steepness of weight curve")
    parser.add_argument("--skew", type=float, required=False, default=0.5, help="Skew of the weight curve (between 0 and 1)")
    args = parser.parse_args()

    #############################

    # Ok it's crazy I have three different df's lol
    df_master = get_formatted_students_sites_table(args.definition_file, args.input_file)
    df = df_master.copy()

    #############################

    df = drop_names(df)
    df = weight_input(df, args.steepness, args.skew)
    df, original_name_dict = resolve_column_names(df)
    indices = do_munkres(df)
    prep_output(df, df_master, original_name_dict, indices)

    #############################

    df_master.to_excel(args.output_file, index=False)

    
if __name__ == "__main__":
    main()
