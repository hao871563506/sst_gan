import git
import click

from src.srgan import SRGAN
from src.utils import addDateTime

@click.command()
@click.option("--job_name", default="test", help="job name, which would be the predix of the result direcotry")
@click.option("--dataset_name", default='hr_lr_timestep_10', help="the name of the dataset")
@click.option("--upscale_power_factor", default=2, help="2^upscale_power_factor = upscale factor. "
                                                        "For example, if you want to upscale by 32, "
                                                        "then the upscale_power_factor should be set to 5.")
@click.option("--n_residual_blocks", default=4, help="number of the resdiual blocks in the generator network")
@click.option("--epochs", default=100, help="number of training intervals")
@click.option("--batch_size", default=1, help="batch size")
@click.option("--sample_interval", default=1680, help="the interval of performing prediction")
@click.option("--root_rslt_dir", default="sst_rslts", help="result diectory")
@click.option("--local", is_flag=True, help="whether to run codes locally")
def main(job_name, dataset_name, upscale_power_factor, n_residual_blocks, epochs, batch_size,
         sample_interval, root_rslt_dir, local):
    # creating directory for saving the prediction plots
    repo = git.Repo('.', search_parent_directories=True)
    repo_dir = repo.working_tree_dir  # sst_supverresolution
    sample_rslt_dir = repo_dir + "/sst_srgan/{}/{}".format(root_rslt_dir, job_name)
    sample_rslt_dir = addDateTime(sample_rslt_dir)

    if local:
        local_dataset_path = repo_dir + "/sst_srgan/datasets/"
    else:
        local_dataset_path = '/rigel/ocp/users/zw2533/sst_superresolution/sst_srgan/numpy_array'
    gan = SRGAN(dataset_name=dataset_name, upscale_power_factor=upscale_power_factor,
                n_residual_blocks=n_residual_blocks, local=local, local_path=local_dataset_path)

    # train
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval, sample_rslt_dir=sample_rslt_dir)

    print("Finish!")


if __name__ == "__main__":
    main()

