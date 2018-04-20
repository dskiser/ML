import java.util.Random;

class MyRandom {
	static Random d = new Random(12349);
	
	static double getdouble() {
		return d.nextDouble();
	}
	
	static int getinteger(int i) {
		return d.nextInt(i);
	}
	
	static double getgaussian() {
		return d.nextGaussian();
	}
}
