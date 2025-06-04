import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.StringTokenizer;
 
public class Main {
	static FastReader fr;
	static int arrForIndexSort[];
	
	
	
	
	
	
	public static void main(String[] args) {
		fr = new FastReader();
		int T = 1;
		T = fr.nextInt();
		int t1 = T;
		while (T-- > 0) {
			solve();
		}
	}
	public static void solve() {
		int n=fr.nextInt();
		int k=fr.nextInt();
		int[] a = new int[k];
		int[] map = new int[n];
		int[] temp = new int[n];
		Arrays.fill(temp, Integer.MAX_VALUE);
		Arrays.fill(map, Integer.MAX_VALUE);
		int min = Integer.MAX_VALUE;
		int max = Integer.MIN_VALUE;
		for(int i=0;i<k;i++) {
			a[i] = fr.nextInt()-1;
			min = Math.min(a[i], min);
			max = Math.max(a[i], max);
		}
		for(int i=0;i<k;i++) {
			map[a[i]] = fr.nextInt();
		}
		int diff = 0;
		int curr = map[min];
		for(int i=min;i<n;i++) {
			int airConditioner = map[i];
			if(airConditioner<curr+diff) {
				temp[i] = airConditioner;
				diff = 1;
				curr = airConditioner;
			}
			else {
				temp[i] = curr+diff;
				diff++;
			}
			
		}
		diff = 0;
		curr = map[max];
		for(int i=max;i>=0;i--) {
			int airConditioner = map[i];
			if(airConditioner<curr+diff) {
				temp[i] = Math.min(temp[i], airConditioner);
				diff = 1;
				curr = airConditioner;
			}
			else {
				temp[i] =  Math.min(temp[i], curr+diff);
				diff++;
			}
			
		}
		System.out.println(Arrays.toString(temp).replaceAll(",|\\[|\\]",""));
		
	}

}
