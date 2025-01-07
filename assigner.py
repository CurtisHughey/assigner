#!/usr/bin/env python3

# pip3 install munkres pandas numpy regex argparse matplotlib XlsxWriter>=3.0.6

# Binder link: https://mybinder.org/v2/gh/CurtisHughey/assigner/HEAD?labpath=assigner.ipynb

from munkres import Munkres, print_matrix

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
import argparse
import os
import xlsxwriter  # Probably don't actually have to import lol
import random

OUTPUT_PREPEND = "OUTPUT_"

PROG_NAME = "ASSIGNER"
MAJOR_VERSION = 1
MINOR_VERSION = 3
PATCH_NUMBER = 0

DEFAULT_STEEPNESS = 3
DEFAULT_SKEW = 0.5


def weight_function(x, a, b, maximum):
    return maximum / (1 + ((maximum/x)**a - 1) ** b)


def calc_midpoint(c, m):
    return 1 / (math.log2(m/c))


# Right now this is not interactive in Jupyter... that seems hard
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
    
    plt.show()  # This just doesn't seem to work in Jupyter... ok
    
    return skew_slider.val, steepness_slider.val


def drop_names(df):
    names = df[df.columns[0]].values
    df = df.drop(df.columns[0], axis=1)
    return df


def weight_input(df, b, skew):
    maximum = df.max(axis=1)[0]  # Might want all the max's...

    a, b = graph_function(b, skew, maximum)

    curried_function = lambda x: weight_function(x, a, b, maximum) 

    df = df.apply(curried_function, axis=1)
    
    return df
    

# TODO I need to comment these functions
def resolve_column_names(df):
    original_name_dict = {}

    multiplier_fmt = r"\(x\d+\)"
    for c in df:
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

    m = Munkres()
    indices = m.compute(matrix)

    total = 0
    for r, c in indices:
        total += master_matrix[r][c]

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
        # Now we look up in matrix for the corresponding row and column. Then we take that column and look up in the master_df (we had created column copies)
        column = df.columns[c]

        actual_column_name = original_name_dict.get(column, column)  # Attempt a lookup, otherwise it was unchanged

        df_output.loc[r, actual_column_name] = "Choice: {}".format(df_copy[actual_column_name].iloc[r])

    return df_output


# Gets the table of students and sites in the format I like:
# Students on x axis
# Sites on y axis. Sites are short name, and have number of slots in parentheses
# Each intersection has the numeric ranking of the student for that site
# HOWEVER. denylist_filename will override a student's ranking. If a student is denied a site, their ranking for that site will be overridden, and instead will be provided a special maximum value (TODO)
def get_formatted_students_sites_table(definitions_filename, input_filename, denylist_filename):

    data = {"Student": []}  # List of names, each of which is a list of preferences
    denylist_dict = parse_denylist(denylist_filename)

    ####################################################################

    # Reads the definitions file (columns of name, long name (survey name), and number of slots), returns a tuple of dicts: name_lookup which allows you to look up the short name from the long name, and preferences_lookup, which allows you to look up the formatted column of slots we want for the next step
    df_definitions = pd.read_excel(definitions_filename)
    
    preferences_lookup = {}  # Points the long name to the short name with the (optional) number of slots
    for name, slots in zip(df_definitions["Name"], df_definitions["Slots"]):
        name = name.strip()
        if int(slots) == 1:
            key = name
        else:
            key = "{} (x{})".format(name, int(slots))
        key = "   {}   ".format(key)  # Adding space before and after... XlsxWriter's autofit doesn't work awesome
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
            if preference in denylist_dict.get(name, []):  # Then this student isn't allowed to have this preference
                ranking = len(preferences)  # We just make it the max preference... I guess this is ok... it could still get assigned...
            else:
                ranking = idx + 1  # idx is 0-based, convert to start at 1
        
            data[preferences_lookup[preference]].append(ranking)
        
    df = pd.DataFrame(data)
    
    return df


# Returns a dict of the denylist
def parse_denylist(denylist_filename):
    if not denylist_filename:
        return {}
    
    # Just two columns: names and then a semicolon-separated list of the (short) names of the places to deny the names to
    df_denylist = pd.read_excel(denylist_filename)

    deny_student_dict = {}
    
    for name, deny_sites in zip(df_denylist["Name"], df_denylist["Deny Sites (semicolon-separated)"]):
        name = name.strip()
        if name in deny_student_dict:
            print("{} appears multiple times in the denylist spreadsheet... I'll combine the denylist but that might not be the behavior you want!".format(name))
        
        existing_deny_sites = deny_student_dict.get(name, [])  # Really arguable that we should be accommodating this...
        new_deny_sites = [x.strip() for x in deny_sites.split(";")]
        
        deny_student_dict[name] = existing_deny_sites + new_deny_sites

    return deny_student_dict


def get_filename(required_string, description, flag, required=True):
    filename = ""
    dirlist = os.listdir()
    for f in dirlist:
        if os.path.isfile(f) and required_string in f.lower() and ".xlsx" in f.lower():
            filename = f
            print("No {} file explicitly provided w/ {}, using {}".format(description, flag, filename))
            break
    else:
        print("Couldn't figure out which {} file to use. Make sure it is uploaded and then provide the name".format(description))
        while True:
            if required:
                filename = input("Enter the {} file name: ".format(description))
            else:
                print("This file is not required. Press <Enter> below if you don't want to pass one in")
                filename = input("Enter the {} file name (<Enter> for none): ".format(description))
                if not filename:  # If nothing was passed in, that's ok, we're done. But otherwise, we'll treat it as normal
                    break
            
            if os.path.isfile(filename):
                break
            else:
                print("Could not find {}, try again".format(filename))
            
    return filename


def shuffle_students(df):
    temp = list(df.index)
    original_permutation = temp[:]
    permutation = temp
    random.shuffle(permutation)
    df = df.reindex(index=permutation)

    return df, original_permutation


def unshuffle_students(df, original_permutation):
    df = df.reindex(index=original_permutation)
    
    return df


def main():
    import socket
    hostname = socket.gethostname()
    
    print("{} {}.{}.{}".format(PROG_NAME, MAJOR_VERSION, MINOR_VERSION, PATCH_NUMBER))
    
    print("Hostname: {}".format(hostname))
    running_in_jupyter = False
    if "jupyter" in hostname:
        print("Running in a Jupyter context!!")
        running_in_jupyter = True
    else:
        print("Probably running in a normal command-line context")
        running_in_jupyter = False

    print("="*50)

    #############################

    # See if we're running in a command line context and that if we should attempt to parse command-line arguments
    if not running_in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-file", type=str, required=False, default="", help="Input file")
        parser.add_argument("--definition-file", type=str, required=False, default="", help="Definitions file")
        parser.add_argument("--output-file", type=str, required=False, default="", help="Output file")
        parser.add_argument("--denylist-file", type=str, required=False, default="", help="Denylist file")
        parser.add_argument("--steepness", type=int, required=False, default=DEFAULT_STEEPNESS, help="Steepness of weight curve")
        parser.add_argument("--skew", type=float, required=False, default=DEFAULT_SKEW, help="Skew of the weight curve (between 0 and 1)")
        args = parser.parse_args()
    else:
        # Otherwise, set it to a bunch of nones
        args = argparse.Namespace()
        args.input_file = ""
        args.definition_file = ""
        args.output_file = ""
        args.denylist_file = ""
        args.steepness = DEFAULT_STEEPNESS
        args.skew = DEFAULT_SKEW

    # If the input file, definition file, and/or output file weren't provided, try to figure out the default. If can't figure out, prompt the user
    if not args.definition_file:
        args.definition_file = get_filename("definition", "definition", "--definition-file")
        print("="*10)
    if not args.input_file:
        args.input_file = get_filename("preferences", "input", "--input-file")
        print("="*10)
    if not args.output_file:
        args.output_file = "{}{}".format(OUTPUT_PREPEND, args.input_file)
        print("Choosing default output file name {}".format(args.output_file))
        print("="*10)
    if not args.denylist_file:
        args.denylist_file = get_filename("denylist", "denylist", "--denylist-file", False)  # This may return an empty string if the user decides there should be no denylist file
        print("="*10)

    print("="*50)
    print("Running program...")
                
    #############################

    df_master = get_formatted_students_sites_table(args.definition_file, args.input_file, args.denylist_file)
    df_master, original_permutation = shuffle_students(df_master)
    df = df_master.copy()
    
    #############################

    df = drop_names(df)
    df = weight_input(df, args.steepness, args.skew)
    df, original_name_dict = resolve_column_names(df)
    indices = do_munkres(df)
    prep_output(df, df_master, original_name_dict, indices)  # Modifies df_master
    df_master = unshuffle_students(df_master, original_permutation)

    #############################
    
    with pd.ExcelWriter(args.output_file) as writer:
        sheet_name = "Results"
        df_master.to_excel(writer, sheet_name = sheet_name, index=False, freeze_panes=(1,1))
        writer.sheets[sheet_name].autofit()

    print("="*50)
    print("Success! Output is in {}".format(args.output_file))
    
if __name__ == "__main__":
    main()

