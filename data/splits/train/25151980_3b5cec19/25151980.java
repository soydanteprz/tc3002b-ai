import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

public class C {
	
	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		int T=fs.nextInt();
		for (int tt=1; tt<=T; tt++) {
			int n=fs.nextInt();
			int m=fs.nextInt();
			int x=fs.nextInt();
			int arr[]=fs.readArray(n);
			if(m>n) {
				System.out.println("NO");
				continue;
			}
			int copy[]=new int [n];
			for(int i=0;i<n;i++) {
				copy[i]=arr[i];
			}
			Arrays.sort(copy);
			ArrayList<Integer> container[]= new ArrayList[x+1];
			for(int i=1;i<=x;i++) {
				container[i]=new ArrayList<Integer>();
			}
			for(int i=0;i<m;i++) {
				for(int j=i;j<n;j+=m) {
					container[copy[j]].add(i+1);
				}
			}
			int counter[]=new int [n];
			for(int i=0;i<n;i++) {
				counter[i]=container[arr[i]].get(0);
				container[arr[i]].remove(0);
			}
			System.out.println("YES");
			for(int i=0;i<n;i++) {
				System.out.print(counter[i]+" ");
			}
			System.out.println();
		}
	}

	

	
}
