import logging
import threading
import pandas as pd
import time


def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)


def Tasks(index):
    df1 = pd.read_csv(f'file{index}.csv', index_col=0)
    for ind in df1.CPEMatchString:
        cpe = ind
        with urllib.request.urlopen(
                "https://services.nvd.nist.gov/rest/json/cpes/1.0/?cpeMatchString={}&addOns=cves&ResultsPerPage=2000".format(
                        cpe)) as url:
            data = json.loads(url.read().decode())
            print(data)
            print('hi')


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    threads = list()
    for index in range(25):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=Tasks(index), args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)