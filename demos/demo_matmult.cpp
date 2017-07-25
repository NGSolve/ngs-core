// c++ -ffp-contract=fast -O3 -mavx2 -mfma -std=c++14 -I../src -L../src demo_matmult.cpp -lngs_core


#include <ngs_core.hpp>
using namespace ngstd;
using namespace ngbla;

// matrix A:  4 x n
// matrix B:  n x 8
// matrix C:  4 x 8
void MatMult_Kernel4x8 (double * pa, size_t da,
                        double * pb, size_t db,
                        double * pc, size_t dc,
                        size_t n)
{
  SIMD<double> sum11(0.0), sum12(0.0);
  SIMD<double> sum21(0.0), sum22(0.0);
  SIMD<double> sum31(0.0), sum32(0.0);
  SIMD<double> sum41(0.0), sum42(0.0);

  __assume(n > 0);
  for (size_t i = 0; i < n; i++, pa++, pb += db)
    {
      SIMD<double> b1(pb);      // load 4 values from address
      SIMD<double> b2(pb+4);   

      SIMD<double> a1(pa[0]);     // load 1 value and broadcast
      sum11 += a1 * b1;
      sum12 += a1 * b2;

      SIMD<double> a2(pa[da]);
      sum21 += a2 * b1;
      sum22 += a2 * b2;

      SIMD<double> a3(pa[2*da]);
      sum31 += a3 * b1;
      sum32 += a3 * b2;

      SIMD<double> a4(pa[3*da]);
      sum41 += a4 * b1;
      sum42 += a4 * b2;
    }

  sum11.Store(pc);
  sum12.Store(pc+4);
  pc += dc;
  sum21.Store(pc);
  sum22.Store(pc+4);
  pc += dc;  
  sum31.Store(pc);
  sum32.Store(pc+4);
  pc += dc;  
  sum41.Store(pc);
  sum42.Store(pc+4);
}


// matrix A:  4 x n
// matrix B:  n x 12
// matrix C:  4 x 12
void MatMult_Kernel4x12 (double * pa, size_t da,
                         double * pb, size_t db,
                         double * pc, size_t dc,
                         size_t n)
{
  SIMD<double> sum11(0.0), sum12(0.0), sum13(0.0);
  SIMD<double> sum21(0.0), sum22(0.0), sum23(0.0);
  SIMD<double> sum31(0.0), sum32(0.0), sum33(0.0);
  SIMD<double> sum41(0.0), sum42(0.0), sum43(0.0);

  __assume(n > 0);  
  for (size_t i = 0; i < n; i++, pa++, pb += db)
    {
      SIMD<double> b1(pb);     // load 4 values from address
      SIMD<double> b2(pb+4);   
      SIMD<double> b3(pb+8);   

      SIMD<double> a1(pa[0]);     // load 1 value and broadcast
      sum11 += a1 * b1;
      sum12 += a1 * b2;
      sum13 += a1 * b3;

      SIMD<double> a2(pa[da]);
      sum21 += a2 * b1;
      sum22 += a2 * b2;
      sum23 += a2 * b3;      

      SIMD<double> a3(pa[2*da]);
      sum31 += a3 * b1;
      sum32 += a3 * b2;
      sum33 += a3 * b3;      

      SIMD<double> a4(pa[3*da]);
      sum41 += a4 * b1;
      sum42 += a4 * b2;
      sum43 += a4 * b3;      
    }

  sum11.Store(pc);
  sum12.Store(pc+4);
  sum13.Store(pc+8);
  pc += dc;
  sum21.Store(pc);
  sum22.Store(pc+4);
  sum23.Store(pc+8);  
  pc += dc;  
  sum31.Store(pc);
  sum32.Store(pc+4);
  sum33.Store(pc+8);  
  pc += dc;  
  sum41.Store(pc);
  sum42.Store(pc+4);
  sum43.Store(pc+8);
}




void MatMult (SliceMatrix<double> a,
              SliceMatrix<double> b,
              BareSliceMatrix<double> c)
{
  // treat main part using optimzed kernels.
  // left over must be handled separately
  
  /*
  for (size_t i = 0; i <= a.Height()-4; i += 4)b
    for (size_t j = 0; j <= b.Width()-8; j += 8)
      {
        MatMult_Kernel4x8(&a(i,0), a.Dist(),
                          &b(0,j), b.Dist(),
                          &c(i,j), c.Dist(), a.Width());
      }
  */

  for (size_t i = 0; i <= a.Height()-4; i += 4)
    for (size_t j = 0; j <= b.Width()-12; j += 12)
      {
        MatMult_Kernel4x12(&a(i,0), a.Dist(),
                           &b(0,j), b.Dist(),
                           &c(i,j), c.Dist(), a.Width());
      }
}
              


int main ()
{
  Timer t("matmult");
  size_t n = 96; // multiple of 12 (resp 8)

  Matrix<double> a(n,n), b(n,n), c(n,n);
  a = 1;
  b = 2;

  size_t ops = n*n*n;   // counting fma as 1 operation
  size_t loops = 1e10 / ops;

  t.Start();
  for (size_t i = 0; i < loops; i++)
    MatMult (a, b, c);
  t.Stop();
  t.AddFlops(ops*loops);
  
  cout << "c(0:10,0:10) = " << c.Rows(0,10).Cols(0,10) << endl;
  
  NgProfiler::Print(stdout);
  // my 2.7 GHz Core i5 gives: 
  // job 1048575 calls        1, time 0.5147 sec, MFlops = 19426.01 matmult
}
