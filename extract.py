#!/usr/bin/env python3

import re
import json
import os
from argparse import ArgumentParser
from io import StringIO
from html.parser import HTMLParser

from logzero import logger
from futil.tokenize import tokenize


global invalid_xml
invalid_tags = re.compile(r"<\s*/?\s*(?:s|unk)\s*>", re.IGNORECASE)
invalid_xml = re.compile('[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')


excluded_start = ["pov", "int.", "ext.", "continued", "later", "fade", "angle", "os.", "sfx",
                  "on ", "at ", "vfx", "cut ", "the end", "the screen", "go to ", "to ",
                  "card:", "with:", "the situation:", ":", "close up", "int ", "ext ", "by the",
                  "across the", "in the", "down the", "into the", "over the", "above the",
                  "outside", "i'm", "yes", "no", "what", "why", "how ", "however", "when", "where", "you ",
                  "you're", "close on ", "a moment later", "still later", "back in", "back at",
                  "in front of", "over at", "through the", "moments later", "finally",
                  "under the", "busy street", "a bit later", "a little later", "a minute later", ]
excluded_tokens = ["int.", "ext.", "draft", "--", "omit", "evening", "morning", "afternoon",
                   "angle", "snapshot", "scene", "**", "dissolve", "continued", "pov",
                   "minutes", "hours", "hour", "month", "months", "shot", "closeup", "season", "episode",
                   "omitted", "cu", "c.u.", "c/u"]

exclude_phrases = ["cut to", "over shot", "end of screenplay", "part i", "part v",
                   "part x", "?", "!", "scene i", "scene v", "scene x", "..."]

tokens_to_delete = ["continued", "v.o.", "o/s", "p.o.v.", "pov"]

turn_tokens = ["my", "i", "you", "your", "me", "you're", "mine", "our", "ours", "us", "don't"]

scene_pattern = r'(?:\b(?:(?:int|ext)\.?|new scene: |interior|exterior|scene [0-9]+)\b|={80,})'
scene_pre_ballast = r'(\s|[0-9]+[ABCDEFGH]?\.?)*'


# Sentence-ending markers (strong and weak)
stop_punctuations1 = ['!', '?']
stop_punctuations2 = ['.', ':', ';']


# constants
MIN_SCENES = 5  # min. number of scenes
MIN_PLAINTEXT = 3000  # min. character limit for file length


class FileTooSmallError(Exception):
    pass

class CantOpenFileError(Exception):
    pass


class TagStripper(HTMLParser):
    """Simple class to strip all tags from HTML while preserving the rest.
    Source: https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python"""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()
        self.should_end_p_with_newline = False

    def handle_starttag(self, tag, attrs):
        # handling some HTML style formats -- essentially converting them to text
        if tag == 'p' and any([a[0] == 'id' for a in attrs]):
            id_attr = [a for a in attrs if a[0] == 'id'][0][1]
            if id_attr == 'speaker':
                self.text.write('\n')
                self.text.write(' ' * 30)
                self.should_end_p_with_newline = True
            elif id_attr == 'dia':
                self.text.write(' ' * 15)
                self.should_end_p_with_newline = True
        # handling TBBT scripts
        elif tag == 's':
            self.text.write('\nNEW SCENE: ')

    def handle_endtag(self, tag):
        if tag == 'p' and self.should_end_p_with_newline:
            self.text.write('\n')

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    """Easy way to call the TagStripper class on a text."""
    s = TagStripper()
    s.feed(html)
    return s.get_data()


class Metadata:

    def __init__(self, filename_mapping, metadata_dir):
        self.mapping = {}
        if filename_mapping:
            with open(filename_mapping, 'r', encoding='UTF-8') as fh:
                data = [line.split("\t") for line in fh.readlines()]
                self.mapping = {item[0].strip(): item[1].strip() for item in data}
        self.metadata_dir = metadata_dir

    def get(self, filename):
        logger.debug('Searching metadata for %s' % filename)
        name = re.sub(r'\.(txt|html|htm)$', '', filename, re.I)
        if self.mapping.get(name):  # mapping existing and non-empty
            logger.debug('Found %s in mapping: %s' % (name, self.mapping[name]))
            metadata_file = os.path.join(self.metadata_dir, self.mapping[name] + '.json')
            if os.path.isfile(metadata_file):
                logger.debug('Metadata file %s exists, loading.' % metadata_file)
                with open(metadata_file, 'r', encoding='UTF-8') as fh:
                    return json.load(fh)
            else:
                return {'imdbID': 'tt' + self.mapping[name]}
        return None


class MovieScript:

    def __init__(self, script_file, do_tokenize=False, do_lowercase=False, metadata=None):

        self.do_tokenize = do_tokenize
        self.do_lowercase = do_lowercase
        self.script_file = script_file
        self.metadata = metadata
        self.title = self._get_title(script_file)
        self.medium = self._detect_medium(script_file)
        self.genre = None
        self.year = None
        logger.info("Opening script \"%s\"" % self.title)

        if script_file.endswith(".html"):
            self.lines = self._convert_to_text(script_file)
        else:
            raise RuntimeError("Script format not supported: " + script_file)

        self.empty_previous_line = True
        self.empty_next_line = False
        self.minimum_space = 20
        self.upper = True
        self.column_style = False  # simpler format, with "Char name: Line" on a single line
        self.scenes_shifted_right = False  # scene names can be a bit more to the right

        self.characters = None
        self._detect_format()
        self.author = self._get_author()
        if self.metadata:
            self.extract_metadata()
        self.extract_dialogues()
        self.postprocess()
        logger.info("-------------")

    def extract_metadata(self):
        if 'seriesTitle' in self.metadata and 'Title' in self.metadata:
            if 'Episode' in self.metadata:
                if 'Season' in self.metadata:
                    self.title = '%s %sx%s: %s' % (self.metadata['seriesTitle'], self.metadata['Season'],
                                                   ('%02d' % int(self.metadata['Episode'])), self.metadata['Title'])
                else:
                    self.title = '%s %s: %s' % (self.metadata['seriesTitle'], self.metadata['Episode'], self.metadata['Title'])
        else:
            self.title = self.metadata.get('Title', self.title)
        self.author = self.metadata.get('Writer', self.author)
        self.genre = self.metadata.get('Genre')
        self.year = self.metadata.get('Year')

    def _get_title(self, script_file):
        title = os.path.basename(script_file)[0:-5].replace("-", " ").lower().strip()
        if title.endswith(", The"):
            title = "The " + title[0:-5]
        elif title.endswith(", A"):
            title = "a " + title[0:-3]
        elif title.endswith(", An"):
            title = "An " + title[0:-4]

        title = re.sub(r'^fd_', '', title, re.I)
        title = re.sub(r'_([0-9]+x[0-9]+)_', r' \1: ', title, re.I)
        return title

    def _detect_medium(self, script_file):
        if re.search('([0-9]+x[0-9]+)', script_file) or re.search('(friends|tbbt)-[0-9]+', script_file):
            return 'TV'
        return 'movie'

    def _get_author(self):
        possible_name = False
        for line in self.lines[:100]:

            if possible_name and re.match('[a-z+]', line.strip(), re.I):
                return line.strip()

            m = re.match(r'^\s*(?:written |story )?by(.*)$', line, re.I)
            if m:
                possible_name = m.group(1).strip()
                if re.match('[a-z]+', possible_name, re.I):
                    return possible_name
                possible_name = True
        return None

    def _detect_format(self):
        min_count = os.path.getsize(self.script_file) / 1000
        logger.debug("Minimum count: " + str(min_count))
        characters = self._extract_characters()
        total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count and "fd_" in self.script_file:
            logger.debug("We need to recheck...")
            self.upper = False
            self.column_style = True
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing with less space...")
            self.minimum_space = 10
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing without necessary empty line before character name")
            self.minimum_space = 20
            self.empty_previous_line = False
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing without indent")
            self.minimum_space = 0
            self.empty_previous_line = True
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing with empty line after character name")
            self.minimum_space = 20
            self.empty_next_line = True
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing without indent and empty lines around")
            self.minimum_space = 0
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing without upper characters")
            self.minimum_space = 20
            self.empty_previous_line = True
            self.empty_next_line = False
            self.upper = False
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing with column format")
            self.upper = True

            self.column_style = True
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])
        if total_count < min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "reparsing with column format and no uppercase")
            self.upper = False
            characters = self._extract_characters()
            total_count = sum([characters[c]["count"] for c in characters])

        if total_count >= min_count:
            logger.debug("Found %i characters (total of %i occurrences), " % (len(characters), total_count)
                         + "final format: %s" % self.get_format())
            sorted_names = sorted(list(characters.keys()), key=lambda c: characters[c]["count"], reverse=True)
            names_list = ["%s (%i)" % (c, characters[c]["count"]) for c in sorted_names]
            names_list = ", ".join(names_list).encode("latin-1", errors="ignore").decode("latin-1")
            logger.debug("Character names: " + names_list)
            self.characters = characters
        else:
            raise RuntimeError("Could not detect format for \"%s\"" % self.title)

        # checking scene formats
        num_scenes = sum(int(self.is_scene_intro(line, idx)) for idx, line in enumerate(self.lines))
        logger.debug("Found %d scenes using default settings." % num_scenes)
        if num_scenes < MIN_SCENES and not self.column_style:
            self.scenes_shifted_right = True  # try with more lenient column format
            num_scenes = sum(int(self.is_scene_intro(line, idx)) for idx, line in enumerate(self.lines))
            logger.debug("Tried right-shifted scene names, found %d scenes." % num_scenes)
        if num_scenes < MIN_SCENES:
            self.scenes_shifted_right = False  # didn't work



    def is_possible_name(self, line, index):
        if not self.column_style:
            min_space = self.minimum_space if not self.characters else 0
            match = re.match(r"\s{%i,40}([^\(\[]+)" % min_space, line)
        else:
            match = re.match(r"\s*(.+?)\:", line)
        if not match:
            return False
        elif self.upper and not match.group(1).isupper():
            return False
        midfeld = match.group(1).lower().strip()
        if len(midfeld) > 35:
            return False
        elif not self.upper and len(midfeld)>25:
            return False

        prev_line = self.lines[index - 1].strip() if index > 0 else ""
        next_line = self.lines[index + 1].strip() if index < len(self.lines) else ""
        # Removing page numbers
        if re.match(r"(\d+[\.\s\:]*)+$", midfeld):
            return False
        elif len(midfeld) > 30:
            return False
        splits = midfeld.split(" ")
        if len(splits) == 2 and splits[0] == splits[1] and re.search(r"\d", splits[0]) and re.search(r"\d", splits[1]):
            return False
        elif len(splits) == 3 and splits[0] == splits[2] and re.search(r"\d", splits[0]) and re.search(r"\d", splits[2]):
            return False
        elif len(splits) == 4 and splits[0] == splits[3] and re.search(r"\d", splits[0]) and re.search(r"\d", splits[3]):
            return False
        # Tokens to avoid in names
        elif any([e for e in excluded_start if midfeld.startswith(e)]):
            return False
        elif any([e for e in excluded_tokens for t in midfeld.split(" ") if e == t]):
            return False
        elif any([e for e in exclude_phrases if e in midfeld]):
            return False
        if not self.column_style and not self.characters:
            if self.empty_previous_line and prev_line:
                return False
            elif not self.empty_next_line and not next_line:
                return False
            elif not prev_line and not next_line and midfeld == self.title:
                return False

        char_name = self.clean_name(line)
        if not char_name or len(char_name) < 2:
            return False
        elif self.characters and char_name not in self.characters:
            return False

        return True

    def is_scene_intro(self, line, index):
        """Checking if the current line is starting a new scene."""
        m = re.search(scene_pattern, line, re.I)
        if m and re.match(scene_pre_ballast, line[:m.start()], re.I):
            if m.start() <= self.minimum_space:
                return True
            elif self.scenes_shifted_right:
                return True
        return False

    def clean_name(self, line):
        if not self.column_style:
            char_name = re.match(r"\s*([^\(\[]+)", line).group(1).title()
        else:
            char_name = re.match(r"\s*(.+?)\:", line).group(1).title()
        char_name = char_name.strip("_")
        tokens = [t for t in char_name.split() if t not in tokens_to_delete]
        cleaned = " ".join(tokens).replace(":", "")
        return cleaned

    def _convert_to_text(self, html_file):
        try:
            fd = open(html_file, encoding="UTF-8")
            fulltext = fd.read()
        except Exception:
            try:
                fd = open(html_file, encoding="cp1252")
                fulltext = fd.read()
            except Exception:
                try:
                    fd = open(html_file)
                    fulltext = fd.read()
                except Exception as e:
                    raise CantOpenFileError("Can't open file %s: %s" % (html_file, str(e)))
        fulltext = re.sub("</?blockquote>", "", fulltext, flags=re.I)
        if len(fulltext) < MIN_PLAINTEXT:
            raise FileTooSmallError("Warning: script %s is suspiciously small: %i characters" % (html_file, len(fulltext)))
        fd.close()
        if "fd_" in self.script_file:
            fulltext = "\n".join([l + "<br>" for l in fulltext.split("\n")])
        text = strip_tags(fulltext)
        text = re.sub("\t", "        ", text)

        lines = text.split('\n')
        return lines

    def get_format(self):
        if self.column_style:
            return "column-style format"
        else:
            str = ("%i minimum spacing, " % self.minimum_space
                   + ("no" if not self.empty_previous_line else "one") + " empty previous line "
                   + "and " + ("no" if not self.empty_next_line else "one") + " empty next line")
            return str

    def _extract_characters(self):
        logger.info("Extracting characters for \"%s\"" % (self.title))
        characters = {}
        name_found = False
        for i, l in enumerate(self.lines):
            if name_found and l.strip() and not self.column_style:
                name_found = False
                continue
            if self.is_possible_name(l, i):
                char_name = self.clean_name(l)
                if char_name not in characters:
                    characters[char_name] = {"count": 1, "original": set([l.strip()])}
                else:
                    characters[char_name]["count"] += 1
                    characters[char_name]["original"].add(l.strip())
                name_found = True

        for c in list(characters.keys()):
            original = list(characters[c]["original"])[0]
            if (characters[c]["count"]< 5
                    and (re.search(r"[\:\-\)]$", original) or re.search(r"\d{2, 5}", original))):
                del characters[c]
        return characters

    def is_end_of_turn(self, l, i, current_turn):
        stripped = l.strip()
        if not stripped:
            return True
        elif self.empty_previous_line and self.is_possible_name(l, i):
            return True

        tokens = stripped.lower().split()
        next_tokens = self.lines[i + 1].strip().lower().split() if i < len(self.lines) else []
        likely_end_of_turn = False
        if tokens[0] == current_turn["character"]:
            likely_end_of_turn = True
        elif len(stripped) - len(current_turn["lines"][-1]) > 8:
            likely_end_of_turn = True
        elif len(current_turn["lines"]) > 1 and len(stripped) - len(current_turn["lines"][-1]) > 4:
            likely_end_of_turn = True
        if (likely_end_of_turn and not any([e for e in turn_tokens for t in tokens if e == t])
                and stripped[0].isupper() and not re.search("[!?]", stripped)
                and not any([e for e in turn_tokens for t in next_tokens if e == t])):
            return True
        return False

    def extract_dialogues(self):
        logger.info("Extracting dialogues for \"%s\"" % (self.title))
        scenes = []
        turns = []
        turns_no = 0

        current_turn = None
        for i, l in enumerate(self.lines):
            stripped = re.sub("[*_]", "", l.strip())
            if current_turn:
                if current_turn["lines"] and self.is_end_of_turn(l, i, current_turn):
                    turns.append(current_turn)
                    turns_no += 1
                    current_turn = None
                if current_turn and stripped and not re.match(r"\(.*\)", stripped) and not re.match(r"[\d\s\.]+$", l):
                    current_turn["lines"].append(stripped)
            if not current_turn and self.is_possible_name(l, i):
                stripped2 = stripped.lower()
                if ("continued" in stripped2 or "cont'" in stripped2) and turns and "character" in turns[-1]:
                    current_turn = turns[-1]
                    turns = turns[:-1]
                else:
                    current_turn = {"character": self.clean_name(l), "lines": []}
                if self.column_style:
                    current_turn["lines"].append(l.split(":")[1].strip())

            elif not current_turn and self.is_scene_intro(l, i):
                # TODO multi-line scene descriptions?
                if turns:
                    scenes.append(turns)
                setting = l.strip()
                setting = re.sub("NEW SCENE:\s*", "", setting)
                turns = [{"setting": setting}]

        scenes.append(turns)  # finish last scene
        self.dialogues = scenes  # save the result

        logger.info('Got %d turns in %d scenes' % (turns_no, len(scenes)))
        if len(scenes) < MIN_SCENES:
            logger.warn('Very few scenes in %s (%d)' % (self.title, len(scenes)))

    def _remove_bracketed(self, text):
        """Remove bracketed stuff -- remarks, emotions etc."""
        text = re.sub(r'\([^\)]*\)', r'', text)
        text = re.sub(r'\[[^\]]*\]', r'', text)
        return text

    def postprocess(self):
        if not self.dialogues:
            raise RuntimeError("Dialogues were not extracted")

        for dialogue in self.dialogues:
            # we're changing the list so we need to use indexes here
            for turn in dialogue:
                if "character" not in turn:
                    continue
                turn['character'] = self._remove_bracketed(turn['character'])
                turn['lines'] = self._remove_bracketed(" ".join(turn["lines"]))
                if self.do_tokenize:
                    turn['lines'] = tokenize(turn['lines'])
                if self.do_lowercase:
                    turn['lines'] = turn['lines'].lower()

    def write_json(self, filename):

        data = {
            'title': {
                'en': self.title,
            },
            'author': self.author,
            'medium': self.medium,
            'character_list': [{'name': name} for name in sorted(self.characters.keys())],
            'scripts': [{
                'language': 'en',
                'source': 'nr.no',
                'acts': [{
                    'scenes': [],
                }]
            }]
        }
        data['character_count'] = len(data['character_list'])
        if self.genre:
            data['genre'] = self.genre
        if self.year:
            data['year'] = self.year
        if self.metadata:
            data['omdb_meta'] = self.metadata
        scenes_list = data['scripts'][0]['acts'][0]['scenes']
        for dialogue in self.dialogues:
            scene = {}
            if 'setting' in dialogue[0]:
                scene['setting'] = dialogue[0]['setting']
                dialogue = dialogue[1:]
            scene['contents'] = [{'character': t['character'], 'text': t['lines']} for t in dialogue]
            scenes_list.append(scene)
        with open(filename, 'w', encoding='UTF-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=4)


def process_movies(args):
    meta = Metadata(args.imdb_mapping, args.omdb_data)
    problems = []
    no_meta = []
    few_scenes = []
    too_small = []
    cant_open = []
    ok = []
    listf = sorted(os.listdir(args.scripts_dir), key=lambda f: 0 if f.startswith("fd_") else 1)
    for f in listf:
        try:
            file_ok = True  # assuming OK
            metadata = meta.get(f)
            if args.imdb_mapping and not meta:
                no_meta.append(f)
                file_ok = False
            ms = MovieScript(os.path.join(args.scripts_dir, f),
                             do_tokenize=args.tokenize, do_lowercase=args.lowercase,
                             metadata=metadata)
            if not ms.column_style and len(ms.dialogues) < MIN_SCENES:
                few_scenes.append(f)
                file_ok = False
            ms.write_json(os.path.join(args.output_dir, os.path.split(ms.script_file)[-1] + '.json'))
            if file_ok and not ms.column_style:
                ok.append(f)

        except FileTooSmallError as e:
            logger.warn(str(e))
            too_small.append(f)

        except CantOpenFileError as e:
            logger.error(str(e))
            cant_open.append(f)

        except RuntimeError as e:
            logger.error("===> Problem with " + f + ": " + str(e))
            if os.path.getsize(os.path.join(args.scripts_dir, f)) > 500:
                problems.append(f)

    logger.info("Processed %d files." % len(listf))
    logger.warn("Couldn't open %d files: %s" % (len(cant_open), ", ".join(cant_open)))
    logger.warn("%d files were too small: %s" % (len(too_small), ", ".join(too_small)))
    logger.warn("Found %d problematic files: %s" % (len(problems), ", ".join(problems)))
    logger.warn("Couldn't get OMDb metadata for %d files: %s" % (len(no_meta), ", ".join(no_meta)))
    logger.warn("Very few scenes in %d non-column files: %s" % (len(few_scenes), ", ".join(few_scenes)))
    logger.info("%d files has no problems (metadata and scene bounds): %s" % (len(ok), ", ".join(ok)))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-H', '--history-length', type=int, default=0,
                    help='History length to prepend before each utterance')
    ap.add_argument('-l', '--lowercase', action='store_true', help='Lowercase all outputs?')
    ap.add_argument('-t', '--tokenize', action='store_true', help='Tokenize all outputs?')
    ap.add_argument('-i', '--imdb-mapping', help='File with mapping to IMDB IDs')
    ap.add_argument('-o', '--omdb-data', help='Directory with JSON metada scraped from OMDb')
    ap.add_argument('scripts_dir', type=str, help='Input movie script directory')
    ap.add_argument('output_dir', type=str, help='Output directory')

    args = ap.parse_args()
    process_movies(args)
