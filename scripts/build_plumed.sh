set -e
export MAMBA_ROOT_PREFIX=/opt/micromamba
eval "$(/opt/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tnfmd
micromamba install -y -c conda-forge make cxx-compiler gsl fftw >/dev/null 2>&1
cd /opt && rm -rf plumed-src && mkdir plumed-src && cd plumed-src
# PLUMED 2.9.x source; --enable-modules=all gives drr (eABF) and opes, which the
# conda-forge build omits.
curl -sL -o plumed.tgz https://github.com/plumed/plumed2/releases/download/v2.9.2/plumed-src-2.9.2.tgz
tar xzf plumed.tgz && cd plumed-2.9.2
./configure --prefix=/opt/plumed292 --enable-modules=all --disable-python --enable-openmp \
  CXXFLAGS="-O3" > conf.log 2>&1 || { tail -30 conf.log; exit 1; }
make -j"$(nproc)" > make.log 2>&1 || { tail -40 make.log; exit 1; }
make install > install.log 2>&1
echo "=== built ==="
/opt/plumed292/bin/plumed info --version
for a in DRR EXTENDED_LAGRANGIAN OPES_METAD METAD COM DISTANCE; do
  n=$(/opt/plumed292/bin/plumed --no-mpi manual --action X 2>&1 | grep -c "^  ${a}$")
  echo "  ${a}: ${n}"
done
