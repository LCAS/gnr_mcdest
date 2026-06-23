import numpy as np
import datetime
import io

import functools


def raise_(e: Exception) -> None:
    """
    Raise an error e. Made for lambdas
    """
    raise e

def print_duck(finished_training):
    if finished_training:
        print(" ____________________ ") # Made using cowsay
        print("< Finished training! >") # echo "Finished training!" | cowsay -f duck
        print(" -------------------- ")
        print(" \\                   ")
        print("  \\                  ")
        print("   \\ >()_            ")
        print("      (__)__ _        ")
    else:
        print(" ____         ")
        print("< ?! >        ")
        print(" ----         ")
        print(" \\           ")
        print("  \\          ")
        print("   \\ >()_    ")
        print("      (__)__ _")

def get_current_iso_datetime() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()

def linesplit_string(long_string: str, line_length=80, pad=True):
    """Splits a long string into an array of lines, padded to min(len(long_string), line_length)

    Args:
        long_string: The string to split.
        line_length: The maximum length of each line.
        pad: Pad or not

    Returns:
        A list of strings, where each string is a line from the original string.
    """
    lines = []
    for i in range(0, len(long_string), line_length):
        line = long_string[i:i + line_length]

        if (pad):
            # Only pad if there are multiple lines
            padding_needed = (line_length - len(line)) if (len(long_string) >= line_length) else 0
            
            if padding_needed > 0:
                line += " " * padding_needed


        lines.append(line)
    
    # Finally, we need to account for user-added EOL    
    lines = functools.reduce(
        lambda x,y: x + y, # Add the arrays
        [ ([line] if (not '\n' in line) else line.split('\n')) for line in lines])

    max_line_len = np.max(np.array([len(line) for line in lines]))

    if pad:
        # Then, we need to re-add padding
        lines = [ line + ' ' * (max_line_len - len(line)) for line in lines ]

        return lines
    
    return lines

def box_outline_text(text: str, border: str = "=", complete_box = True) -> str:
    # Wrap lines
    lines = linesplit_string(text)
    # Get longest line, and go from there
    line_length = np.max([ len(line) for line in lines ]) + (4 if complete_box else 0)
    borders = border * line_length

    out = borders

    for line in lines:
        if complete_box:
            out += "\n" + border + " " + line + " " + border
        else:
            out += "\n" + line

    return out + "\n" + borders

def print_box_outline_text(text: str, border: str = "=") -> None:
    print(box_outline_text(text, border))


def normalise_for_cmap(arr, cid):
    # Normalize the array to 0-1 range
    min_val = np.min(arr)
    max_val = np.max(arr)


    # Select one cat
    arr = arr[cid]
    # Can't normalise zerores
    if (np.sum(arr) == 0):
        return arr
        
    normalised_arr = (arr - min_val) / (max_val - min_val)
    # Scale to uint8
    normalised_arr = (normalised_arr * 255).astype(np.uint8)

    return normalised_arr

def fig_to_numpy(fig):
    """https://stackoverflow.com/a/67823421"""
    
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    
    return data.reshape((int(h), int(w), -1))[:,:,:3]



# "Unit test" lol
# print_box_outline_text("Hello, world!")
# print_box_outline_text("Hello,\nworld!")
# print_box_outline_text("Hello,\nworld!\nWhy what a long message this is.\nTest123\nefbhbefeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\ntest\nrrgrgrhrjhgbrhjhjb\nhebfhjebjebhfebhfbejbeheb\n\n\nWow.. glad that worked?")
# print_box_outline_text("Helloooooo,\nworld!")