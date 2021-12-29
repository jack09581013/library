COLOR_RESET = '\033[0m'


def rgb(r, g, b, background=False):
    """
    print(rgb(255, 128, 0)
      + rgb(80, 30, 60, True)
      + 'Fancy colors!'
      + COLOR_RESET)

      see https://stackoverflow.com/questions/45782766/color-python-output-given-rrggbb-hex-value
    """
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)


def rgb_hex(hex_string, background=False):
    if hex_string[0] == '#':
        hex_string = hex_string[1:]
    assert len(hex_string) == 6, 'Hex string should contain 6 Hexadecimal numbers'

    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)

    return rgb(r, g, b, background=background)
