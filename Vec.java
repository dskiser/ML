// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Iterator;
import java.lang.StringBuilder;

/// Represents a vector of doubles
public class Vec
{
	protected double[] vals;
	protected int start;
	protected int len;

	/// Makes a vector of the specified size
	public Vec(int size)
	{
		if(size == 0)
			vals = null;
		else
			vals = new double[size];
		start = 0;
		len = size;
	}

	/// Wraps the specified array of doubles
	public Vec(double[] data)
	{
		vals = data;
		start = 0;
		len = data.length;
	}

	/// This is NOT a copy constructor. It wraps the same buffer of values as v.
	public Vec(Vec v, int begin, int length)
	{
		vals = v.vals;
		start = v.start + begin;
		len = length;
	}

	/// Unmarshalling constructor
	public Vec(Json n)
	{
		vals = new double[n.size()];
		for(int i = 0; i < n.size(); i++)
			vals[i] = n.getDouble(i);
		start = 0;
		len = n.size();
	}
	
	/// Creates weight Vec from weight Matrix.
	public Vec(int numInter, Matrix M) // intercept must be first row
	{
		len = M.rows() * M.cols();
		vals = new double[len];
		
		int k = 0;
		// insert intercept values into Vec
		for(int i=0; i<M.cols(); i++) vals[k++] = M.get(0, i);
		
		// insert remaining values into Vec
		for(int i=0; i<M.cols(); i++) {
			for(int j=1; j<M.rows(); j++) {
				vals[k++] = M.get(j,i);
			}
		}
	}

	public Json marshal()
	{
		Json list = Json.newList();
		for(int i = 0; i < len; i++)
			list.add(vals[start + i]);
		return list;
	}

	public int size()
	{
		return len;
	}
	
	// Returns a double[] array version of Vec
	double[] array;
	public double[] toArray() {
		array = new double[this.size()];
		for(int i = 0; i < this.size(); i++) 
			array[i] = this.get(i);
		return array;
	}
		

	public double get(int index)
	{
		return vals[start + index];
	}

	public void set(int index, double value)
	{
		vals[start + index] = value;
	}

	public void fill(double val)
	{
		for(int i = 0; i < len; i++)
			vals[start + i] = val;
	}

	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		if(len > 0)
		{
			sb.append(Double.toString(vals[start]));
			for(int i = 1; i < len; i++)
			{
				sb.append(",");
				sb.append(Double.toString(vals[start + i]));
			}
		}
		return sb.toString();
	}

	public double squaredMagnitude()
	{
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += vals[start + i] * vals[start + i];
		return d;
	}

	public void normalize()
	{
		double mag = squaredMagnitude();
		if(mag <= 0.0) {
			fill(0.0);
			vals[0] = 1.0;
		} else {
			double s = 1.0 / Math.sqrt(mag);
			for(int i = 0; i < len; i++)
				vals[i] *= s;
		}
	}

	public void copy(Vec that)
	{
		vals = new double[that.size()];
		for(int i = 0; i < that.size(); i++)
			vals[i] = that.get(i);
		start = 0;
		len = that.size();
	}

	public void add(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < len; i++)
			vals[start + i] += that.get(i);
	}
	
	int new_length;
	Vec combined;
	public Vec attach(Vec that) {
		new_length = this.size() + that.size();
		combined = new Vec(new_length);
		
		int k = 0;
		// Insert first Vec into combined Vec
		for(int i = 0; i < this.size(); i++) {
			combined.set(k++, this.get(i));
		}
		// Insert second Vec into combined Vec
		for(int i = 0; i < that.size(); i++) {
			combined.set(k++, that.get(i));
		}
		
		return combined;
	}
	
	public void scale(double scalar)
	{
		for(int i = 0; i < len; i++)
			vals[start + i] *= scalar;
	}

	public void addScaled(double scalar, Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < len; i++)
			vals[start + i] += scalar * that.get(i);
	}
	
	protected Vec sum;
	public Vec sum(double scalar, Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		sum = new Vec(that.size());
		for(int i = 0; i < that.size(); i++) {
			sum.set(i, ((scalar*that.get(i)) + this.get(i)));
		}
		return sum;
	}
	
	public boolean equal(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < that.size(); i++) {
			if(that.get(i) != this.get(i))
				return false;
		}
		return true;
	}
	
	/// Returns the index of the max value in Vec
	public int maxIndex()
	{
		int index = 0;
		double value = 0.0;
		for(int i = 0; i < len; i++) {
			double new_value = get(i);
			if(new_value > value) {
				index = i;
				value = new_value;
			}
		}
		return index;
	}
	
	/// Returns the sum of values in Vec
	protected double summation;
	public double summation() 
	{
		Vec ones = new Vec(this.size());
		ones.fill(1.0);
		summation = this.dotProduct(ones);
		
		return summation;
	}

	public double dotProduct(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += get(i) * that.get(i);
		return d;
	}
	
	public Matrix outerProduct(Vec that) {
		Matrix op = new Matrix(0, that.size());
		for(int n = 0; n < this.size(); n++) {
			double[] row = new double[that.size()];
			for(int p = 0; p < that.size(); p++) {
				row[p] = this.get(n) * that.get(p);
			}
			op.takeRow(row);
		}
		return op;
	}
	double d, t;
	public double squaredDistance(Vec that)
	{
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		d = 0.0;
		for(int i = 0; i < len; i++)
		{
			t = get(i) - that.get(i);
			d += (t * t);
		}
		return d;
	}
	
	Vec original;
	public void reverse() {
		original = new Vec(0);
		original.copy(this);
		for(int i = 0; i < this.size(); i++) {
			this.set(i, original.get(this.size() - i - 1));
		}
	}	
}

	






/// A tensor class.
class Tensor extends Vec
{
	int[] dims;

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(Vec vals, int[] _dims)
	{
		super(vals, 0, vals.size());
		dims = new int[_dims.length];
		int tot = 1;
		for(int i = 0; i < _dims.length; i++)
		{
			dims[i] = _dims[i];
			tot *= _dims[i];
		}
		if(tot != vals.size())
			throw new RuntimeException("Mismatching sizes. Vec has " + Integer.toString(vals.size()) + ", Tensor has " + Integer.toString(tot));
	}

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(Tensor copyMe)
	{
		super((Vec)copyMe, 0, copyMe.size());
		dims = new int[copyMe.dims.length];
		for(int i = 0; i < copyMe.dims.length; i++)
			dims[i] = copyMe.dims[i];
	}

	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	protected static int[] kinner, kouter, stepInner, stepFilter, stepOuter;
	static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride)
	{
		// Precompute some values
		int dc = in.dims.length;
		if(dc != filter.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		if(dc != out.dims.length)
			throw new RuntimeException("Expected tensors with the same number of dimensions");
		kinner = new int[dc];
		kouter = new int[dc];
		stepInner = new int[dc];
		stepFilter = new int[dc];
		stepOuter = new int[dc];

		// Compute step sizes
		stepInner[0] = 1;
		stepFilter[0] = 1;
		stepOuter[0] = 1;
		for(int i = 1; i < dc; i++)
		{
			stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
			stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
			stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
		}
		int filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

		// Do convolution
		int op = 0;
		int ip = 0;
		int fp = 0;
		for(int i = 0; i < dc; i++)
		{
			kouter[i] = 0;
			kinner[i] = 0;
			int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
		}
		while(true) // kouter
		{
			double val = 0.0;

			// Fix up the initial kinner positions
			for(int i = 0; i < dc; i++)
			{
				int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				int adj = (padding - Math.min(padding, (int)kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}
			while(true) // kinner
			{
				val += (in.get(ip) * filter.get(flipFilter ? filterTail - fp : fp));

				// increment the kinner position
				int i;
				for(i = 0; i < dc; i++)
				{
					kinner[i]++;
					ip += stepInner[i];
					fp += stepFilter[i];
					int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
					if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
						break;
					int adj = (padding - Math.min(padding, (int)kouter[i])) - kinner[i];
					kinner[i] += adj;
					fp += adj * stepFilter[i];
					ip += adj * stepInner[i];
				}
				if(i >= dc)
					break;
			}
			out.set(op, out.get(op) + val);

			// increment the kouter position
			int i;
			for(i = 0; i < dc; i++)
			{
				kouter[i]++;
				op += stepOuter[i];
				ip += stride * stepInner[i];
				if(kouter[i] < out.dims[i])
					break;
				op -= kouter[i] * stepOuter[i];
				ip -= kouter[i] * stride * stepInner[i];
				kouter[i] = 0;
			}
			if(i >= dc)
				break;
		}
	}

	/// Throws an exception if something is wrong.
	static void test()
	{
		{
			// 1D test
			Vec in = new Vec(new double[]{2,3,1,0,1});
			Tensor tin = new Tensor(in, new int[]{5});

			Vec k = new Vec(new double[]{1, 0, 2});
			Tensor tk = new Tensor(k, new int[]{3});

			Vec out = new Vec(7);
			Tensor tout = new Tensor(out, new int[]{7});

			Tensor.convolve(tin, tk, tout, true, 1);

			//     2 3 1 0 1
			// 2 0 1 --->
			Vec expected = new Vec(new double[]{2, 3, 5, 6, 3, 0, 2});
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}

		{
			// 2D test
			Vec in = new Vec(new double[]
				{
					1, 2, 3,
					4, 5, 6,
					7, 8, 9
				}
			);
			Tensor tin = new Tensor(in, new int[]{3, 3});

			Vec k = new Vec(new double[]
				{
					1,  2,  1,
					0,  0,  0,
					-1, -2, -1
				}
			);
			Tensor tk = new Tensor(k, new int[]{3, 3});

			Vec out = new Vec(9);
			Tensor tout = new Tensor(out, new int[]{3, 3});

			Tensor.convolve(tin, tk, tout, false, 1);
			
			Vec expected = new Vec(new double[]
				{
					-13, -20, -17,
					-18, -24, -18,
					13,  20,  17
				}
			);
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}
	}
}
