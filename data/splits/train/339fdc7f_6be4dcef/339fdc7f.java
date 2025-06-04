import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

public class B{
	static FastScanner sc = new FastScanner();
	static BufferedWriter output = new BufferedWriter(
			new OutputStreamWriter(System.out));
	
	
	public static long util(int arr[]) {
		Arrays.sort(arr);
		int i = 0;
		int j = arr.length - 1;
		while(i <= j) {
			int tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
		int n = arr.length;
		//min priority queue
		PriorityQueue<Integer> pq = new PriorityQueue<>();
		long ans = 0;
		long sum = 0;
		long cnt = 0;
		for(int ii: arr) {
			if( ii == 0) cnt++;
		}
		for(i =0;i<n;i++) {
			
			sum += arr[i];
			ans++;
			if(arr[i] < 0) {
				pq.add(arr[i]);
			}
			if( sum <= 0 ) {
				while( sum <= 0 ) {
					if(pq.isEmpty()) break;
					sum -= pq.peek();
					ans--;
					pq.remove();
					
				}
			}
		}
		if(sum > 0) 
			return ans;
		return ans - cnt;
		
		
		
	}
	
	
	public static void solve() {
		int n = sc.nextInt();
		String a[] = new String[n];
		for(int i =0;i<n;i++) {
			a[i] = sc.next();
		}
		
		long ans =  -1;
		for(int i =0;i<5;i++) {
			char mx = (char)(i + 'a');
			int c1 = 0, c2 = 0;
			int arr[] = new int[n];
			for(int j =0;j<n;j++) {
				c1 = 0;
				c2 = 0;
				for(int k = 0;k<a[j].length();k++) {
					if(a[j].charAt(k) == mx) c1++;
					else c2++;
				}
				arr[j]= c1 - c2;
			}
			ans = Math.max(ans, util(arr));
		}
		System.out.println(ans);
		
	}
	public static void main(String[] args) {
		int t = 1;
		t = sc.nextInt();
		outer: for (int tt = 0; tt < t; tt++) {
			solve();
		}
	}
	
////////////////////////////////////////////////////////////////////////////////////
////////////////////DO NOT BELOW THIS LINE //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
	
	
	
	
	
	
	
	
	
	
////////////////////////////////////////////////////////////////////////////////////	
	
}
