import java.io.*;
import java.util.*;
public class Pupsen {
	public static void main(String[] args) throws Exception {
		FastIO in = new FastIO();
		int t = in.nextInt();
		for (int tc=0; tc<t; tc++) {
			int n = in.nextInt();
			int[] a = new int[n];
			for (int i=0; i<n; i++) {
				a[i] = in.nextInt();
			}
			int[] b = new int[n];
			if (n%2==0) {
				for (int i=0; i<n-1; i+=2) {
					b[i] = -a[i+1];
					b[i+1] = a[i];
				}
				for (int i=0; i<n; i++) System.out.print(b[i]+" ");
			}
			else {
				if (a[0]+a[1]!=0) {
					b[0] = -a[2];
					b[1] = -a[2];
					b[2] = a[0]+a[1];
				}
				else if (a[0]+a[2]!=0) {
					b[0] = -a[1];
					b[2] = -a[1];
					b[1] = a[0]+a[2];
				}
				else {
					b[1] = -a[0];
					b[2] = -a[0];
					b[0] = a[1]+a[2];
				}
				for (int i=3; i<n-1; i+=2) {
					b[i] = -a[i+1];
					b[i+1] = a[i];
				}
				for (int i=0; i<n; i++) System.out.print(b[i]+" ");
			}
			System.out.println();
		}
	}
	
}
