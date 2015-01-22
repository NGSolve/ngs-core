#ifndef FILE_NGBLA_EXPR
#define FILE_NGBLA_EXPR

/**************************************************************************/
/* File:   expr.hpp                                                       */
/* Author: Joachim Schoeberl                                              */
/* Date:   01. Jan. 02                                                    */
/**************************************************************************/

#ifdef USE_GMP
#include <gmpxx.h>
#endif


namespace ngbla
{
  
  // not yet functional
  enum ORDERING { ColMajor, RowMajor };

  template <typename T = double, ORDERING ORD = RowMajor> class FlatMatrix;
  template <typename T = double, ORDERING ORD = RowMajor> class Matrix;

  template <int H, int W, typename T> class Mat;
  template <int H, typename T> class DiagMat;
  template <int S, typename T> class Vec;


  /*
    Matrix expression templates
  */


  template <typename TVEC> 
  inline typename TVEC::TELEM & Access (TVEC & vec, int nr)
  {
    return vec(nr);
  }

  template <typename TVEC> 
  inline typename TVEC::TELEM Access (const TVEC & vec, int nr)
  {
    return vec(nr);
  }

  inline double & Access (double & vec, int nr)
  {
    return vec;
  }

  inline double Access (const double & vec, int nr)
  {
    return vec;
  }

  inline Complex & Access (Complex & vec, int nr)
  {
    return vec;
  }

  inline Complex Access (const Complex & vec, int nr)
  {
    return vec;
  }






  template <typename TM> 
  inline typename TM::TELEM & Access (TM & mat, int i, int j)
  {
    return mat(i,j);
  }

  template <typename TM> 
  inline typename TM::TELEM Access (const TM & mat, int i, int j)
  {
    return mat(i,j);
  }

  inline double & Access (double & mat, int i, int j)
  {
    return mat;
  }

  inline double Access (const double & mat, int i, int j)
  {
    return mat;
  }

  inline Complex & Access (Complex & mat, int i, int j)
  {
    return mat;
  }

  inline Complex Access (const Complex & mat, int i, int j)
  {
    return mat;
  }










  /**
     Trait to obtain vector and scalar types for given matrix types.
     Is specified for double, Complex, AutoDiff<doube>, AutoDiff<Complex>
  */




  /*
  template <class T>
  class mat_traits
  {
  public:
    /// matrix element
    typedef typename T::TELEM TELEM;
    /// field of matrix element
    typedef typename T::TSCAL TSCAL;
    /// type of column vector
    typedef typename T::TV_COL TV_COL;
    /// type of row vector
    typedef typename T::TV_ROW TV_ROW;
    /// matrix height
    enum { HEIGHT = T::HEIGHT };
    /// matrix with
    enum { WIDTH  = T::WIDTH  };
    ///
    enum { IS_COMPLEX = mat_traits<TSCAL>::IS_COMPLEX };
  };
  */

  template <class T>  class mat_traits;

  template <class T>
  class mat_traits
  {
  public:
    /// matrix element
    typedef T TELEM;
    /// field of matrix element
    typedef T TSCAL;
    /// type of column vector
    typedef T TV_COL;
    /// type of row vector
    typedef T TV_ROW;
    /// matrix height
    enum { HEIGHT = 1 };
    /// matrix with
    enum { WIDTH  = 1  };
    ///
    enum { IS_COMPLEX = 0 };
  };

  template <class T>
  class mat_traits<const T> : public mat_traits<T> { };
  template <class T>
  class mat_traits<T&> : public mat_traits<T> { };


  template <int D>
  class mat_traits<ngstd::INT<D> >
  {
  public:
    typedef int TELEM;
    typedef int TSCAL;
    typedef int TV_COL;
    typedef int TV_ROW;
    enum { HEIGHT = D };
    enum { WIDTH = 1 };
    enum { IS_COMPLEX = 0 };
  };

  template <>
  class mat_traits<Complex>
  {
  public:
    typedef Complex TELEM;
    typedef Complex TSCAL;
    typedef Complex TV_COL;
    typedef Complex TV_ROW;
    enum { HEIGHT = 1 };
    enum { WIDTH = 1 };
    enum { IS_COMPLEX = 1 };
  };


  template <int D, typename SCAL>
  class mat_traits<AutoDiff<D,SCAL> >
  {
  public:
    typedef AutoDiff<D,SCAL> TELEM;
    typedef AutoDiff<D,SCAL> TSCAL;
    typedef AutoDiff<D,SCAL> TV_COL;
    typedef AutoDiff<D,SCAL> TV_ROW;
    enum { HEIGHT = 1 };
    enum { WIDTH = 1 };
    enum { IS_COMPLEX = mat_traits<SCAL>::IS_COMPLEX };
  };









  /// Height of matrix
  template <class TM> 
  inline int Height (const TM & m)
  {
    return m.Height();
  }

  /// Width of matrix
  template <class TM> 
  inline int Width (const TM & m)
  {
    return m.Width();
  }

  template <> inline int Height<double> (const double&) { return 1; }
  template <> inline int Height<Complex> (const Complex&) { return 1; }
  template <> inline int Width<double> (const double&) { return 1; }
  template <> inline int Width<Complex> (const Complex&) { return 1; }


  /// Complex to double assignment called
  class Complex2RealException : public Exception
  {
  public:
    Complex2RealException()
      : Exception("Assignment of Complex 2 Real") { ; }
  };



  template <typename TO>
  inline TO ConvertTo (double f)
  {
    return TO(f);
  }

  template <typename TO>
  inline TO ConvertTo (Complex f)
  {
    return TO(f);
  }


  template <typename TO>
  inline TO ConvertTo (const AutoDiff<1, Complex> & f)
  {
    return TO(f);
  }


  template <>
  inline double ConvertTo (Complex f)
  {
    throw Complex2RealException();
  }


  template <class TA> class RowsArrayExpr;
  template <class TA> class ColsArrayExpr;
  template <class TA> class SubMatrixExpr;
  template <class TA> class RowExpr;
  template <class TA> class ColExpr;




#ifdef CHECK_RANGE
  /// matrices do not fit for matrix-matrix operation
  class MatrixNotFittingException : public Exception
  {
  public:
    MatrixNotFittingException (const string & where, int h1, int w1,
			       int h2, int w2)
      : Exception ("") 
    {
      stringstream str;
      str << where << " mat1 = " << h1 << " x " << w1 << ", mat2 = " << h2 << " x " << w2 << endl;
      Append (str.str());
    }
  };
#endif


  /**
     Expr is the base class for all matrix template expressions.
     Barton and Nackman Trick for template polymorphism, function Spec.

     provides Height and Width of matrix.
     IsLinear allows linear matrix element access.
  */

  template <typename T>
  class Expr 
  {
  public:
    INLINE Expr () { ; }

    /// cast to specific type
    INLINE T & Spec() { return static_cast<T&> (*this); }

    /// cast to specific type
    INLINE const T & Spec() const { return static_cast<const T&> (*this); }


    /// height 
    INLINE int Height() const { return Spec().T::Height(); }
    INLINE int Width() const { return Spec().T::Width(); }

    // INLINE auto operator() (int i) const -> decltype (this->Spec()(i)) { return this->Spec()(i); }
    // INLINE auto operator() (int i, int j) const -> decltype (this->Spec()(i,j)) { return this->Spec()(i,j); }

    void Dump (ostream & ost) const { Spec().T::Dump(ost); }


    INLINE RowExpr<const T> Row (int r) const
    {
      return RowExpr<const T> (static_cast<const T&> (*this), r);
    }

    INLINE ColExpr<T> Col (int r) const
    {
      return RowExpr<T> (static_cast<T&> (*this), r);
    }


    INLINE SubMatrixExpr<T>
    Rows (int first, int next)
    { 
      return SubMatrixExpr<T> (static_cast<T&> (*this), first, 0, next-first, Width()); 
    }

    SubMatrixExpr<T>
    Cols (int first, int next) 
    { 
      return SubMatrixExpr<T> (static_cast<T&> (*this), 0, first, Height(), next-first);
    }

    SubMatrixExpr<T>
    Rows (IntRange range) 
    { 
      return Rows (range.First(), range.Next());
    }

    SubMatrixExpr<T>
    Cols (IntRange range) 
    { 
      return Cols (range.First(), range.Next());
    }


    RowsArrayExpr<T>
    Rows (FlatArray<int> rows) 
    { 
      return RowsArrayExpr<T> (static_cast<const T&> (*this), rows); 
    }

    ColsArrayExpr<T>
    Cols (FlatArray<int> cols) 
    { 
      return ColsArrayExpr<T> (static_cast<const T&> (*this), cols); 
    }

  };













  /**
     Caller knows that matrix expression is a symmetric matrix.
     Thus, only one half of the matrix needs to be computed.
  */
  template <typename T>
  class SymExpr : public Expr<SymExpr<T> >
  {
    const T & a;
  public:

    SymExpr (const T & aa) : a(aa) { ; }

    INLINE auto operator() (int i) const -> decltype (a(i)) { return a(i); }
    INLINE auto operator() (int i, int j) const -> decltype (a(i,j)){ return a(i,j); }
    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }
    enum { IS_LINEAR = T::IS_LINEAR };
    void Dump (ostream & ost) const
    { ost << "Sym ("; a.Dump(ost); ost << ")"; }
  };


  /**
     Declare that matrix expression is symmetric
  */
  template <typename T>
  inline SymExpr<T> Symmetric (const Expr<T> & a)
  {
    return SymExpr<T> (static_cast<const T&> (a));
  }



  template <typename TA>
  class LapackExpr : public Expr<LapackExpr<TA> >
  {
    const TA & a;
  public:
    LapackExpr (const TA & aa) : a(aa) { ; }
    const TA & A() const { return a; }
    int Height() const { return a.Height(); }
    int Width() const { return a.Width(); }
  };


  enum  T_Lapack { Lapack };
  
  template <typename TA>
  INLINE LapackExpr<TA> operator| (const Expr<TA> & a, T_Lapack /* tl */)
  {
    return LapackExpr<TA> (a.Spec());
  }





  template <typename TA>
  class LocalHeapExpr : public Expr<LocalHeapExpr<TA> >
  {
    const TA & a;
    LocalHeap * lh;
  public:
    INLINE LocalHeapExpr (const TA & aa, LocalHeap & alh) : a(aa), lh(&alh) { ; }
    INLINE const TA & A() const { return a; }
    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }
    INLINE LocalHeap & GetLocalHeap() const { return *lh; }
  };
  
  template <typename TA>
  INLINE LocalHeapExpr<TA> operator| (const Expr<TA> & a, LocalHeap & lh)
  {
    return LocalHeapExpr<TA> (a.Spec(), lh);
  }





  
  template <class TA, class TB> class MultExpr;
  
  
  /**
     The base class for matrices.
  */
  template <class T>
  class MatExpr : public Expr<T>
  {
  public:
 
    INLINE MatExpr () { ; }

    using Expr<T>::Spec;
    using Expr<T>::Height;
    using Expr<T>::Width;

    enum { IS_LINEAR = 1 };  // row-major continuous storage (dist=width)
    enum { COL_MAJOR = 0 };  // matrix is stored col-major

    void Dump (ostream & ost) const { ost << "Matrix"; }




    template<typename TOP, typename TB>
    INLINE T & Assign (const Expr<TB> & v)
    {
#ifdef CHECK_RANGE
      if (Height() != v.Height() || Width() != v.Width())
	{
	  throw MatrixNotFittingException ("operator=", 
					   Height(), Width(),
					   v.Height(), v.Width());
	}
#endif

      if (T::COL_MAJOR)
        {
	  int h = Expr<T>::Height();
	  int w = Expr<T>::Width();

          for (int j = 0; j < w; j++)
            for (int i = 0; i < h; i++)
              TOP()(Spec()(i,j), v.Spec()(i,j));
          return Spec();
        }


      if (TB::IS_LINEAR)
	{
	  if (T::IS_LINEAR)
	    {
	      int hw = Expr<T>::Height() * Expr<T>::Width();
	      for (int i = 0; i < hw; i++)
		TOP()(Spec()(i),v.Spec()(i));
	    }
	  else
	    {
	      int h = Expr<T>::Height();
	      int w = Expr<T>::Width();
	      for (int i = 0, k = 0; i < h; i++)
		for (int j = 0; j < w; j++, k++)
		  TOP() (Spec()(i,j), v.Spec()(k));
	    }
	}
      else
	{
	  int h = Expr<T>::Height();
	  int w = Expr<T>::Width();

	  if (T::IS_LINEAR)
	    for (int i = 0, k = 0; i < h; i++)
	      for (int j = 0; j < w; j++, k++)
		TOP() (Spec()(k), v.Spec()(i,j));
	  else
	    for (int i = 0; i < h; i++)
	      for (int j = 0; j < w; j++)
		TOP() (Spec()(i,j), v.Spec()(i,j));
	}
      return Spec();
    }


    class As 
    {
    public:
      template <typename T1, typename T2> 
      INLINE void operator() (T1 && v1, const T2 & v2) { v1 = v2; }
    };
    class AsAdd 
    {
    public:
      template <typename T1, typename T2> 
      INLINE void operator() (T1 && v1, const T2 & v2) { v1 += v2; }
    };
    class AsSub 
    {
    public:
      template <typename T1, typename T2> 
      INLINE void operator() (T1 && v1, const T2 & v2) { v1 -= v2; }
    };
	

    template<typename TB>
    INLINE T & operator= (const Expr<TB> & v)
    {
      Assign<As> (v);
      return Spec();
    }


    template<typename TB>
    INLINE T & operator+= (const Expr<TB> & v)
    {
      Assign<AsAdd> (v);
      return Spec();
    }

    template<typename TB>
    INLINE MatExpr<T> & operator-= (const Expr<TB> & v)
    {
      Assign<AsSub> (v);
      return Spec();
    }




    template <typename TA, typename TB>
    INLINE T & operator= (const Expr<LapackExpr<MultExpr<TA, TB>>> & prod) 
    {
      LapackMultAdd (prod.Spec().A().A(), prod.Spec().A().B(), 1.0, Spec(), 0.0);
      return Spec();
    }

    template <typename TA, typename TB>
    INLINE T & operator+= (const Expr<LapackExpr<MultExpr<TA, TB> > > & prod)
    {
      LapackMultAdd (prod.Spec().A().A(), prod.Spec().A().B(), 1.0, Spec(), 1.0);
      return Spec();
    }

    template <typename TA, typename TB>
    INLINE T & operator-= (const Expr<LapackExpr<MultExpr<TA, TB> > > & prod)
    {
      LapackMultAdd (prod.Spec().A().A(), prod.Spec().A().B(), -1.0, Spec(), 1.0);
      return Spec();
    }






    template<typename TB>
    INLINE T & operator+= (const Expr<SymExpr<TB> > & v)
    {
#ifdef CHECK_RANGE
      if (Height() != v.Height() || Width() != v.Width())
	throw MatrixNotFittingException ("operator+=", 
					 Height(), Width(),
					 v.Height(), v.Width());
#endif
      int h = Height();
      for (int i = 0; i < h; i++)
	{
	  for (int j = 0; j < i; j++)
	    {
	      double val = v.Spec()(i,j);
	      Spec()(i,j) += val;
	      Spec()(j,i) += val;
	    }
	  Spec()(i,i) += v.Spec()(i,i);
	}
      return Spec();
    }



    template <class SCAL2>
    INLINE T & operator*= (SCAL2 s)
    {
      if (T::IS_LINEAR)
	{
	  int hw = Height() * Width();
	  for (int i = 0; i < hw; i++)
	    Spec()(i) *= s;
	}
      else
	for (int i = 0; i < Height(); i++)
	  for (int j = 0; j < Width(); j++)
	    Spec()(i,j) *= s;
	
      return Spec();
    }

    template <class SCAL2>
    INLINE T & operator/= (SCAL2 s)
    {
      return (*this) *= (1./s);
    }
  };











  /**
     The base class for matrices.
     Constant-Means-Constat-Pointer
     matrix-values may be chaneged by const methods
  */
  template <class T>
  class CMCPMatExpr : public MatExpr<T>
  {
  public:
    // int Height() const { return Spec().T::Height(); }
    // int Width() const { return Spec().T::Width(); }

    INLINE CMCPMatExpr () { ; }

    using MatExpr<T>::Spec;
    using MatExpr<T>::Height;
    using MatExpr<T>::Width;

    // T & Spec() { return static_cast<T&> (*this); }
    // const T & Spec() const { return static_cast<const T&> (*this); }
    // enum { IS_LINEAR = 1 };

    template<typename TB>
    INLINE const T & operator= (const Expr<TB> & v) const
    {
      const_cast<CMCPMatExpr*> (this) -> MatExpr<T>::operator= (v);
      return Spec();
    }

    template<typename TB>
    INLINE const T & operator+= (const Expr<TB> & v) const
    {
      const_cast<CMCPMatExpr*> (this) -> MatExpr<T>::operator+= (v);
      return Spec();
    }

    template<typename TB>
    INLINE const T & operator+= (const Expr<SymExpr<TB> > & v) const
    {
      const_cast<CMCPMatExpr*> (this) -> MatExpr<T>::operator+= (v);
      return Spec();
    }

    template<typename TB>
    INLINE const T & operator-= (const Expr<TB> & v) const
    {
      const_cast<CMCPMatExpr*> (this) -> MatExpr<T>::operator-= (v);
      return Spec();
    }

    template <class SCAL2>
    INLINE const T & operator*= (SCAL2 s) const
    {
      const_cast<CMCPMatExpr*> (this) -> MatExpr<T>::operator*= (s);
      return Spec();
    }

    template <class SCAL2>
    INLINE const T & operator/= (SCAL2 s) const 
    {
      return (*this) *= (1.0/s);
    }

    SubMatrixExpr<const T>
    INLINE Rows (int first, int next) const
    { 
      return SubMatrixExpr<const T> (static_cast<const T&> (*this), first, 0, next-first, Width()); 
    }

    SubMatrixExpr<const T>
    INLINE Cols (int first, int next) const
    { 
      return SubMatrixExpr<const T> (static_cast<const T&> (*this), 0, first, Height(), next-first);
    }

    SubMatrixExpr<const T>
    INLINE Rows (IntRange range) const
    { 
      return Rows (range.First(), range.Next());
    }

    SubMatrixExpr<const T>
    INLINE Cols (IntRange range) const
    { 
      return Cols (range.First(), range.Next());
    }



    RowsArrayExpr<T>
    Rows (FlatArray<int> rows) const
    { 
      return RowsArrayExpr<T> (static_cast<const T&> (*this), rows); 
    }

    ColsArrayExpr<T>
    Cols (FlatArray<int> cols) const
    { 
      return ColsArrayExpr<T> (static_cast<const T&> (*this), cols); 
    }
  };

















  /* *************************** SumExpr **************************** */

  /**
     Sum of 2 matrix expressions
  */

  template <class TA, class TB> 
  class SumExpr : public Expr<SumExpr<TA,TB> >
  {
    const TA & a;
    const TB & b;
  public:

    enum { IS_LINEAR = TA::IS_LINEAR && TB::IS_LINEAR };
    
    INLINE SumExpr (const TA & aa, const TB & ab) : a(aa), b(ab) { ; }

    INLINE auto operator() (int i) const -> decltype(a(i)+b(i)) { return a(i)+b(i); }
    INLINE auto operator() (int i, int j) const -> decltype(a(i,j)+b(i,j)) { return a(i,j)+b(i,j); }

    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }

    void Dump (ostream & ost) const
    { ost << "("; a.Dump(ost); ost << ") + ("; b.Dump(ost); ost << ")"; }
  };

  template <typename TA, typename TB>
  INLINE SumExpr<TA, TB>
  operator+ (const Expr<TA> & a, const Expr<TB> & b)
  {
    return SumExpr<TA, TB> (a.Spec(), b.Spec());
  }




  /* *************************** SubExpr **************************** */


  /**
     Matrix-expr minus Matrix-expr
  */

  template <class TA, class TB> 
  class SubExpr : public Expr<SubExpr<TA,TB> >
  {
    const TA & a;
    const TB & b;
  public:

    enum { IS_LINEAR = TA::IS_LINEAR && TB::IS_LINEAR };
    
    INLINE SubExpr (const TA & aa, const TB & ab) : a(aa), b(ab) { ; }

    INLINE auto operator() (int i) const -> decltype(a(i)-b(i)) { return a(i)-b(i); }
    INLINE auto operator() (int i, int j) const -> decltype(a(i,j)-b(i,j)) { return a(i,j)-b(i,j); }
    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }
  };


  template <typename TA, typename TB>
  INLINE SubExpr<TA, TB>
  operator- (const Expr<TA> & a, const Expr<TB> & b)
  {
    return SubExpr<TA, TB> (a.Spec(), b.Spec());
  }







  /* *************************** MinusExpr **************************** */


  /**
     minus Matrix-expr
  */

  template <class TA>
  class MinusExpr : public Expr<MinusExpr<TA> >
  {
    const TA & a;
  public:
    MinusExpr (const TA & aa) : a(aa) { ; }

    auto operator() (int i) const -> decltype(-a(i)) { return -a(i); }
    auto operator() (int i, int j) const -> decltype(-a(i,j)) { return -a(i,j); }
    int Height() const { return a.Height(); }
    int Width() const { return a.Width(); }

    enum { IS_LINEAR = TA::IS_LINEAR };
  };

  template <typename TA>
  inline MinusExpr<TA>
  operator- (const Expr<TA> & a)
  {
    return MinusExpr<TA> (a.Spec());
  }


  /* *************************** ScaleExpr **************************** */


  /**
     Scalar times Matrix-expr
  */
  template <class TA, class TS> 
  class ScaleExpr : public Expr<ScaleExpr<TA,TS> >
  {
    const TA & a;
    TS s;
  public:
    enum { IS_LINEAR = TA::IS_LINEAR };

    INLINE ScaleExpr (const TA & aa, TS as) : a(aa), s(as) { ; }

    INLINE auto operator() (int i) const -> decltype(s*a(i)) { return s * a(i); }
    INLINE auto operator() (int i, int j) const -> decltype(s*a(i,j)) { return s * a(i,j); }

    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }
    void Dump (ostream & ost) const
    { ost << "Scale, s=" << s << " * "; a.Dump(ost);  }
  };

  template <typename TA>
  INLINE ScaleExpr<TA, double> 
  operator* (double b, const Expr<TA> & a)
  {
    return ScaleExpr<TA, double> (a.Spec(), b);
  }

  template <typename TA>
  INLINE ScaleExpr<TA, Complex> 
  operator* (Complex b, const Expr<TA> & a)
  {
    return ScaleExpr<TA, Complex> (a.Spec(), b);
  }
  
  template <int D, typename TAD, typename TA>
  INLINE ScaleExpr<TA, AutoDiff<D,TAD> > 
  operator* (const AutoDiff<D,TAD> & b, const Expr<TA> & a)
  {
    return ScaleExpr<TA, AutoDiff<D,TAD> > (a.Spec(), b );
  }




  /* ************************* MultExpr ************************* */


  /**
     Matrix-expr timex Matrix-expr
  */
  template <class TA, class TB> class MultExpr : public Expr<MultExpr<TA,TB> >
  {
    const TA & a;
    const TB & b;
  public:

    INLINE MultExpr (const TA & aa, const TB & ab) : a(aa), b(ab) { ; }

    INLINE auto operator() (int i) const -> decltype(a(0,0)*b(0,0))
    { return operator()(i,0); }  

    INLINE auto operator() (int i, int j) const -> decltype (a(0,0)*b(0,0))
    { 
      int wa = a.Width();

      if (wa >= 1)
	{
	  auto sum = a(i,0) * b(0,j);
	  for (int k = 1; k < wa; k++)
	    sum += a(i,k) * b(k,j);
          return sum;
	}

      decltype (a(0,0)*b(0,0)) sum (0);
      return sum;
      // return decltype(a(0,0)*b(0,0)) (0);
      // return 0;
    }

    INLINE const TA & A() const { return a; }
    INLINE const TB & B() const { return b; }
    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return b.Width(); }
    enum { IS_LINEAR = 0 };
  };


  template <int H, typename SCALA, class TB> class MultExpr<DiagMat<H,SCALA>,TB> 
    : public Expr<MultExpr<DiagMat<H,SCALA>,TB> >
  {
    const DiagMat<H,SCALA> & a;
    const TB & b;
  public:

    MultExpr (const DiagMat<H,SCALA> & aa, const TB & ab) : a(aa), b(ab) { ; }

    auto operator() (int i) const -> decltype(a(0,0)*b(0,0))
    { return a[i] * b(i); }  

    auto operator() (int i, int j) const -> decltype (a(0,0)*b(0,0))
    { return a[i] * b(i,j); }

    const DiagMat<H,SCALA> & A() const { return a; }
    const TB & B() const { return b; }
    int Height() const { return a.Height(); }
    int Width() const { return b.Width(); }
    enum { IS_LINEAR = 0 };
  };


  template <typename TA, typename TB>
  INLINE MultExpr<TA, TB>
  operator* (const Expr<TA> & a, const Expr<TB> & b)
  {
    return MultExpr<TA, TB> (a.Spec(), b.Spec());
  }


  /* ************************** Trans *************************** */


  INLINE double Trans (double a) { return a; }
  INLINE Complex Trans (Complex a) { return a; }
  template<int D, typename TAD>
  INLINE AutoDiff<D,TAD> Trans (const AutoDiff<D,TAD> & a) { return a; }


  /**
     Transpose of Matrix-expr
  */
  template <class TA> class TransExpr : public MatExpr<TransExpr<TA> >
  {
    const TA & a;
  public:
    INLINE TransExpr (const TA & aa) : a(aa) { ; }

    INLINE int Height() const { return a.Width(); }
    INLINE int Width() const { return a.Height(); }

    INLINE auto operator() (int i, int j) const -> decltype(Trans (a(j,i))) { return Trans (a(j,i)); }
    INLINE auto operator() (int i) const -> decltype(Trans(a(0,0))) { return 0; }
    // auto Row (int i) const -> decltype (a.Col(i)) { return a.Col(i); }
    // auto Col (int i) const -> decltype (a.Row(i)) { return a.Row(i); }
    enum { IS_LINEAR = 0 };

    INLINE const TA & A() const { return a; }
  };


  /// Transpose 
  template <typename TA>
  INLINE TransExpr<TA>
  Trans (const Expr<TA> & a)
  {
    return TransExpr<TA> (a.Spec());
  }


  /* ************************* SubMatrix ************************ */

  template <class TA> 
  class SubMatrixExpr : public MatExpr<SubMatrixExpr<TA> >
  {
    TA & a;
    int first_row, first_col;
    int height, width;
  public:
    SubMatrixExpr (TA & aa, int fr, int fc, int ah, int aw) 
      : a(aa), first_row(fr), first_col(fc), height(ah), width(aw) { ; }

    int Height() const { return height; }
    int Width() const { return width; }

    auto operator() (int i, int j) -> decltype(a(0,0)) { return a(i+first_row, j+first_col); }
    auto operator() (int i) -> decltype(a(0)) { return a(i+first_row); }
    auto operator() (int i, int j) const -> decltype(a(0,0)) { return a(i+first_row, j+first_col); }
    auto operator() (int i) const -> decltype(a(0)) { return a(i+first_row); }

    enum { IS_LINEAR = 0 };
    enum { COL_MAJOR = TA::COL_MAJOR };

    template<typename TB>
    const SubMatrixExpr & operator= (const Expr<TB> & m) 
    {
      MatExpr<SubMatrixExpr<TA> >::operator= (m);
      return *this;
    }
  };


  template <class TA> 
  class RowExpr : public MatExpr<RowExpr<TA> >
  {
    TA & a;
    int row;
  public:
    RowExpr (TA & aa, int r)
      : a(aa), row(r) { ; }

    int Height() const { return 1; }
    int Width() const { return a.Width(); }

    auto operator() (int i, int j) -> decltype(a(0,0)) { return a(row,i); }
    auto operator() (int i) -> decltype(a(0,0)) { return a(row,i); }
    auto operator() (int i, int j) const -> decltype(a(0,0)) { return a(row,i); }
    auto operator() (int i) const -> decltype(a(0,0)) { return a(row,i); }

    enum { IS_LINEAR = 0 };
    
    template<typename TB>
    const RowExpr & operator= (const Expr<TB> & m) 
    {
      MatExpr<RowExpr<TA> >::operator= (m);
      return *this;
    }
  };






  /* ************************* RowsArray ************************ */

  /**
     RowsArray
  */
  template <class TA> class RowsArrayExpr : public MatExpr<RowsArrayExpr<TA> >
  {
    const TA & a;
    FlatArray<int> rows;
  public:
    // typedef typename TA::TELEM TELEM;
    // typedef typename TA::TSCAL TSCAL;

    RowsArrayExpr (const TA & aa, FlatArray<int> arows) : a(aa), rows(arows) { ; }

    int Height() const { return rows.Size(); }
    int Width() const { return a.Width(); }

    auto operator() (int i, int j) const -> decltype(a(rows[i])) { return a(rows[i], j); }
    auto operator() (int i) const -> decltype(a(rows[i])) { return a(rows[i]); }

    auto Row (int i) const -> decltype (a.Row(rows[i])) { return a.Row(rows[i]); }

    enum { IS_LINEAR = 0 };

    template<typename TB>
    const RowsArrayExpr & operator= (const Expr<TB> & m) 
    {
      MatExpr<RowsArrayExpr<TA> >::operator= (m);
      return *this;
    }

    const RowsArrayExpr & operator= (const RowsArrayExpr & m) 
    {
      MatExpr<RowsArrayExpr<TA> >::operator= (m);
      return *this;
    }
  };
  

  /**
     ColsArray
  */
  template <class TA> class ColsArrayExpr : public MatExpr<ColsArrayExpr<TA> >
  {
    const TA & a;
    FlatArray<int> cols;
  public:
    // typedef typename TA::TELEM TELEM;
    // typedef typename TA::TSCAL TSCAL;

    ColsArrayExpr (const TA & aa, FlatArray<int> acols) : a(aa), cols(acols) { ; }

    int Height() const { return a.Height(); }
    int Width() const { return cols.Size(); }

    // auto operator() (int i, int j) const -> decltype(a(rows[i])) { return a(rows[i], j); }
    // auto operator() (int i) const -> decltype(a(rows[i])) { return a(rows[i]); }

    // TELEM & operator() (int i, int j) const { return a(i, cols[j]); }
    // TELEM & operator() (int i) const { return a(i, cols[0]); }

    auto operator() (int i, int j) const -> decltype(a(i,cols[j])) { return a(i, cols[j]); }
    auto operator() (int i) const -> decltype(a(i,cols[0])) { return a(i, cols[0]); }

    enum { IS_LINEAR = 0 };

    template<typename TB>
    const ColsArrayExpr & operator= (const Expr<TB> & m) 
    {
      MatExpr<ColsArrayExpr<TA> >::operator= (m);
      return *this;
    }

  };
  



  /* ************************* Conjugate *********************** */ 


  INLINE double Conj (double a)
  {
    return a;
  }

  INLINE Complex Conj (Complex a)
  {
    return conj(a);
  }

  /**
     Conjugate of Matrix-expr
  */
  // Attention NOT transpose !!! elemwise conjugate !!! 
  template <class TA> class ConjExpr : public Expr<ConjExpr<TA> >
  {
    const TA & a;
  public:
    // typedef typename TA::TELEM TELEM;
    // typedef typename TA::TSCAL TSCAL;

    INLINE ConjExpr (const TA & aa) : a(aa) { ; }

    INLINE int Height() const { return a.Height(); }
    INLINE int Width() const { return a.Width(); }
 
    INLINE auto operator() (int i, int j) const -> decltype(Conj(a(i,j))) { return Conj(a(i,j)); }
    INLINE auto operator() (int i) const -> decltype(Conj(a(i))) { return Conj(a(i)); }

    enum { IS_LINEAR = 0 };
  };


  /// Conjugate
  template <typename TA>
  INLINE ConjExpr<TA>
  Conj (const Expr<TA> & a)
  {
    return ConjExpr<TA> (static_cast <const TA&> (a));
  }

  template<int D, typename TAD>
  INLINE AutoDiff<D,TAD> Conj (const AutoDiff<D,TAD> & a) 
  { 
    AutoDiff<D,TAD> b; 
    b.Value() = conj(a.Value()); 
  
    for(int i=0;i<D;i++) 
      b.DValue(i) = conj(a.DValue(i)); 

    return b;
  }





  /* ************************* InnerProduct ********************** */


  INLINE double InnerProduct (double a, double b) {return a * b;}
  INLINE Complex InnerProduct (Complex a, Complex b) {return a * b;}
  INLINE Complex InnerProduct (double a, Complex b) {return a * b;}
  INLINE Complex InnerProduct (Complex a, double b) {return a * b;}
  INLINE int InnerProduct (int a, int b) {return a * b;}

  /**
     Inner product
  */

  template <class TA, class TB>
  INLINE auto InnerProduct (const Expr<TA> & a, const Expr<TB> & b)
    -> decltype (InnerProduct(a.Spec()(0), b.Spec()(0)))
  {
    if (a.Height() == 0) return 0; 

    auto sum = InnerProduct (a.Spec()(0), b.Spec()(0));
    for (int i = 1; i < a.Height(); i++)
      sum += InnerProduct (a.Spec()(i), b.Spec()(i));
    return sum;
  }







  /* **************************** Trace **************************** */


  /**
     Calculates the trace of a matrix expression.
  */
  template <class TA>
  inline auto Trace (const Expr<TA> & a) -> decltype (a.Spec()(0,0))
  {
    // typedef typename TA::TELEM TELEM;
    decltype (a.Spec()(0,0)) sum = 0;
    for (int i = 0; i < Height(a); i++)
      sum += a.Spec()(i,i);
    return sum;
  }

  /* **************************** L2Norm **************************** */

  /// Euclidean norm squared
  inline double L2Norm2 (double v)
  {
    return v*v;
  }

  inline double L2Norm2 (Complex v)
  {
    return v.real()*v.real()+v.imag()*v.imag();
  }

  template<int D, typename SCAL>
  inline double L2Norm2 (const AutoDiff<D,SCAL> & x) 
  {
    return L2Norm2(x.Value());
  }


  template <class TA>
  inline double L2Norm2 (const Expr<TA> & v)
  {
    double sum = 0;
    if (TA::IS_LINEAR)
      for (int i = 0; i < v.Height()*v.Width(); i++)
	sum += L2Norm2 (v.Spec()(i));  
    else
      for (int i = 0; i < v.Height(); i++)
	for (int j = 0; j < v.Width(); j++)
	  sum += L2Norm2 (v.Spec()(i,j));  
    
    return sum;
  }

  /**
     Calculates the Euclidean norm
  */
  template <class TA>
  inline double L2Norm (const Expr<TA> & v)
  {
    return sqrt (L2Norm2(v));
  }



  /* **************************** MaxNorm **************************** */

  /// Euclidean norm squared
  inline double MaxNorm (double v)
  {
    return fabs(v);
  }

  inline double MaxNorm (Complex v)
  {
    return fabs(v);
  }

  template<int D, typename SCAL>
  inline double L2Norm (const AutoDiff<D,SCAL> & x) throw()
  {
    return MaxNorm(x.Value());
  }

  template <class TA>
  inline double MaxNorm (const Expr<TA> & v)
  {
    double sum = 0;

    if (TA::IS_LINEAR)
      for (int i = 0; i < v.Height()*v.Width(); i++)
	sum = max(sum, MaxNorm( v.Spec()(i)) );  
    else
      for (int i = 0; i < v.Height(); i++)
	for (int j = 0; j < v.Width(); j++)
	  sum = max(sum, MaxNorm ( v.Spec()(i,j)) );  
    
    return sum;
  }


  /* *************************** Output ****************************** */



  /// Print matrix-expr
  template<typename T>
  ostream & operator<< (ostream & s, const Expr<T> & v)
  { 
    int width = s.width();
    if (width == 0) width = 8;
    s.width(0);
    for (int i = 0; i < v.Height(); i++)
      {
	for (int j = 0 ; j < v.Width(); j++)
	  s << " " << setw(width-1) << v.Spec()(i,j);
	s << endl;
      }
    return s;
  }



  /* **************************** Inverse *************************** */



  /// Calculate inverse. Gauss elimination with row pivoting
  template <class T2>
  extern NGS_DLL_HEADER void CalcInverse (FlatMatrix<T2> inv);

  extern NGS_DLL_HEADER void CalcInverse (FlatMatrix<double> inv);

  template <class T, class T2>
  inline void CalcInverse (const FlatMatrix<T> m, FlatMatrix<T2> inv)
  {
    inv = m;
    CalcInverse (inv);
  }

  template <class T, class T2>
  inline void CalcInverse (const FlatMatrix<T> m, Matrix<T2> & inv)
  {
    inv = m;
    CalcInverse (inv);
  }

  


  /**
     Calculates the inverse of a Matrix.
  */
  template <typename T>
  inline Matrix<T> Inv (const FlatMatrix<T> & m)
  {
    Matrix<T> inv(m.Height(),m.Height());
    CalcInverse (m, inv);
    return inv;
  }

  inline void CalcInverse (double & m)
  {
    m = 1 / m;
  }

  inline void CalcInverse (Complex & m)
  {
    m = 1.0 / m;
  }

  template <int H, int W, typename T>
  inline void CalcInverse (Mat<H,W,T> & m)
  {
    FlatMatrix<T> fm(m);
    CalcInverse (fm);
  }


  INLINE void CalcInverse (const double & m, double & inv)
  {
    inv = 1 / m;
  }

  INLINE void CalcInverse (const Complex & m, Complex & inv)
  {
    inv = 1.0 / m;
  }

  template <int H, int W, typename T, typename TINV>
  inline void CalcInverse (const Mat<H,W,T> & m, TINV & inv)
  {
    FlatMatrix<T> fm(m);
    FlatMatrix<T> finv(inv);
    CalcInverse (fm, finv);
  }

  template <typename T, typename TINV>
  INLINE void CalcInverse (const Mat<0,0,T> & m, TINV & inv)
  {
    ;
  }

  template <typename T, typename TINV>
  INLINE void CalcInverse (const Mat<1,1,T> & m, TINV & inv)
  {
    inv(0,0) = 1.0 / m(0,0);
  }

  template <typename T, typename TINV>
  INLINE void CalcInverse (const Mat<2,2,T> & m, TINV & inv)
  {
    T idet = 1.0 / (m(0,0) * m(1,1) - m(0,1) * m(1,0));
    inv(0,0) = idet * m(1,1);
    inv(0,1) = -idet * m(0,1);
    inv(1,0) = -idet * m(1,0);
    inv(1,1) = idet * m(0,0);
  }

  template <typename T, typename TINV>
  INLINE void CalcInverse (Mat<3,3,T> m, TINV & inv)
  {
    T h0 = m(4)*m(8)-m(5)*m(7);
    T h1 = m(5)*m(6)-m(3)*m(8);
    T h2 = m(3)*m(7)-m(4)*m(6);
    T det = m(0) * h0 + m(1) * h1 + m(2) * h2;
    T idet = 1.0 / det;
    
    inv(0,0) =  idet * h0; 
    inv(0,1) = -idet * (m(1) * m(8) - m(2) * m(7));
    inv(0,2) =  idet * (m(1) * m(5) - m(2) * m(4));
    
    inv(1,0) =  idet * h1; 
    inv(1,1) =  idet * (m(0) * m(8) - m(2) * m(6));
    inv(1,2) = -idet * (m(0) * m(5) - m(2) * m(3));
    
    inv(2,0) =  idet * h2; 
    inv(2,1) = -idet * (m(0) * m(7) - m(1) * m(6));
    inv(2,2) =  idet * (m(0) * m(4) - m(1) * m(3));
    return;
  }



  template <int H, int W, typename T>
  INLINE Mat<H,W,T> Inv (Mat<H,W,T> m)
  {
    Mat<H,W,T> inv;
    CalcInverse (m, inv);
    return inv;
  }

  template <int H, int W, typename T>
  INLINE Mat<H,W,T> Adj (Mat<H,W,T> m)
  {
    cerr << "Adj<" << H << "," << W << "> not implemented" << endl;
    return m;
  }


  template <typename T>
  INLINE Mat<0,0,T> Adj (Mat<0,0,T> m)
  {
    return Mat<0,0,T>();
  }

  template <typename T>
  INLINE Mat<1,1,T> Adj (Mat<1,1,T> m)
  {
    Mat<1,1,T> adj;
    adj(0,0) = m(0,0);
    return adj;
  }

  template <typename T>
  INLINE Mat<2,2,T> Adj (Mat<2,2,T> m)
  {
    Mat<2,2,T> adj;
    adj(0,0) = m(1,1);
    adj(0,1) = -m(0,1);
    adj(1,0) = -m(1,0);
    adj(1,1) = m(0,0);
    return adj;
  }


  template <typename T>
  INLINE Mat<3,3,T> Adj (Mat<3,3,T> m)
  {
    Mat<3,3,T> adj;
    adj(0,0) =  m(4)*m(8)-m(5)*m(7);
    adj(0,1) = -(m(1) * m(8) - m(2) * m(7));
    adj(0,2) =  m(1) * m(5) - m(2) * m(4);
    
    adj(1,0) =  m(5)*m(6)-m(3)*m(8);
    adj(1,1) =  m(0) * m(8) - m(2) * m(6);
    adj(1,2) = -(m(0) * m(5) - m(2) * m(3));
    
    adj(2,0) =  (m(3)*m(7)-m(4)*m(6));
    adj(2,1) = -(m(0) * m(7) - m(1) * m(6));
    adj(2,2) =  (m(0) * m(4) - m(1) * m(3));
    return adj;
  }



  template <int H, int W, typename T>
  INLINE Mat<H,W,T> Cof (Mat<H,W,T> m)
  {
    cerr << "Cof<" << H << "," << W << "> not implemented" << endl;
    return m;
  }

  template <typename T>
  INLINE Mat<0,0,T> Cof (Mat<0,0,T> m)
  {
    return Mat<0,0,T>();
  }

  template <typename T>
  INLINE Mat<1,1,T> Cof (Mat<1,1,T> m)
  {
    Mat<1,1,T> cof;
    cof(0,0) = m(0,0);
    return cof;
  }

  template <typename T>
  INLINE Mat<2,2,T> Cof (Mat<2,2,T> m)
  {
    Mat<2,2,T> cof;
    cof(0,0) = m(1,1);
    cof(0,1) = -m(1,0);
    cof(1,0) = -m(0,1);
    cof(1,1) = m(0,0);
    return cof;
  }


  template <typename T>
  INLINE Mat<3,3,T> Cof (Mat<3,3,T> m)
  {
    Mat<3,3,T> cof;
    cof(0,0) =  m(1,1)*m(2,2)-m(2,1)*m(1,2);
    cof(0,1) = -m(1,0)*m(2,2)+m(2,0)*m(1,2);
    cof(0,2) =  m(1,0)*m(2,1)-m(2,0)*m(1,1);
    
    cof(1,0) = -m(0,1)*m(2,2)+m(2,1)*m(0,2); 
    cof(1,1) =  m(0,0)*m(2,2)-m(2,0)*m(0,2); 
    cof(1,2) = -m(0,0)*m(2,1)+m(2,0)*m(0,1);
    
    cof(2,0) =  m(0,1)*m(1,2)-m(1,1)*m(0,2); 
    cof(2,1) = -m(0,0)*m(1,2)+m(1,0)*m(0,2); 
    cof(2,2) =  m(0,0)*m(1,1)-m(1,0)*m(0,1);
    return cof;
  }






  


  /* ********************** Determinant *********************** */


  /**
     Calculates the determinant of a Matrix.
  */
  template <class T>
  inline typename T::TELEM Det (const MatExpr<T> & m)
  {
    const T & sm = m.Spec();
    switch (sm.Height())
      {
      case 1: 
	{
	  return sm(0,0); 
	}
      case 2:
	{
	  return ( sm(0,0)*sm(1,1) - sm(0,1)*sm(1,0) ); 
	}
      case 3:
	{
	  return 
	    sm(0) * (sm(4) * sm(8) - sm(5) * sm(7)) +
	    sm(1) * (sm(5) * sm(6) - sm(3) * sm(8)) +
	    sm(2) * (sm(3) * sm(7) - sm(4) * sm(6));
	}
      default:
	{
	  cerr << "general det not implemented" << endl;
	}
      }

    return typename T::TELEM (0);  
  }
}

#endif
