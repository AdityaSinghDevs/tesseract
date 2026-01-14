### Note on Result File Naming

Early benchmark runs used a simpler file naming scheme.
After benchmarking logic stabilized, result filenames were standardized to:

{benchmark}_{device}_{config}_batch_size_{N}_{prompt}.json

Along with better logging of parameters.
For consistency, earlier result files were renamed without modifying their contents.
All metrics inside JSON files are produced directly by the benchmark scripts.
