import os

def join_files() :
    # Define the input file and output file names
    input_file = "part_files/fake_news_classifier - Copy.h5"
    output_file = "fake_news_classifier - Copy_2.h5"

    # Define the chunk size (in bytes)
    chunk_size = 24 * 1024 * 1024  # 50 MB

    try:
        with open(output_file, "wb") as out_file:
            part_num = 1
            while True:
                part_name = f"{input_file}.part{part_num:03d}"
                if not os.path.exists(part_name):
                    break
                with open(part_name, "rb") as part_file:
                    out_file.write(part_file.read())
                part_num += 1
        print(f"File parts recombined into '{output_file}' and part files deleted.")
    except Exception as e:
        print(f"Error recombining file parts: {e}")

