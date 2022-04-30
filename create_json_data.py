from extract import MovieScript
import json
import glob
import config

TV_SCRIPTS_DIRECTORY = "/net/projects/THEaiTRE/nrno_movies_scripts"

files_to_extract = glob.glob(f'{config.TV_SCRIPTS_DIRECTORY}/tbbt*.html')


with open("tbbt_dialogues.json", "w") as f:
    for file in files_to_extract:
        script = MovieScript(file)
        print(len(script.dialogues))
        for setting_dialogue in script.dialogues:
            for utterance in setting_dialogue[1:]:
                f.write(f'{json.dumps({"Line":utterance["lines"],"Character":utterance["character"]})}\n')
                # f.write(f'{utterance["lines"]}::{utterance["character"]}\n')
                # file.write('\n')
                # print(utterance)  