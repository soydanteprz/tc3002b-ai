import java.util.*;
import java.io.*;
public class D {
	static ArrayList<Integer> set = new ArrayList<Integer>();
	static int[] a;
	static int test;
	public static void main(String[] args) throws IOException{
		Scanner in = new Scanner(System.in);
		int t = in.nextInt();
		for(int q=0;q<t;q++) {
			int n = in.nextInt();
			a = new int[n];
			for(int i=0;i<n;i++) {
				a[i] = Math.abs(in.nextInt());
			}
			boolean yes = false;
			for(int i=0;i<n;i++) {
				if(a[i]==0) yes = true;
				int[] b = new int[n-1];
				int index=0;
				for(int j=0;j<n;j++) {
					if(j!=i) {
						b[index] = a[j];
						index++;
					}
				}
				test = a[i];
				set = new ArrayList<Integer>();
				if(b.length!=0) {
					if(subset(b,0)) yes = true;					
				}
			}
			if(yes) {
				System.out.println("YES");
			}else {
				System.out.println("NO");
			}
		}
	}
	
	static boolean subset(int[] a,int i) {
//		System.out.println(set);
		if(i>=a.length) {
			if(binary(new int[set.size()],0)) return true;
			return false;
		}
		set.add(a[i]);
		if(subset(a,i+1))return true;
		set.remove(set.size()-1);
		if(subset(a,i+1))return true;
		return false;
	}
	
	static boolean binary(int[] b,int i) {
		if(i==b.length) {
			int sum=0;
			for(int j=0;j<b.length;j++) {
				if(b[j]==0) {
					sum+= set.get(j);
				}else {
					sum-= set.get(j);
				}
			}
//			System.out.println(sum+" "+test);
			if(sum==test) {
				return true;
			}else {
				return false;
			}
		}
		b[i] = 0;
		if(binary(b,i+1))return true;
		b[i] = 1;
		if(binary(b,i+1))return true;
		return false;
	}
	
}
