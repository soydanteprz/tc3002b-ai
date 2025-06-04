import java.util.*;
import java.io.*;
import java.time.*;
import static java.lang.Math.*;

@SuppressWarnings("unused")
public class C {
	static boolean DEBUG = false;

	static Reader fs;
	static PrintWriter pw;

	static void solve() {
		int n = fs.nextInt(), k[] = fs.readArray(n), h[] = fs.readArray(n);
		int prev_h = h[0], prev_k = k[0];
//		int ans = 0;
		ArrayList<pair> intervals = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			int start = k[i] - h[i] + 1;
			int end = k[i];
			intervals.add(new pair(start, end));
		}
//		pw.println(intervals);
		Collections.sort(intervals);
		ArrayList<pair> merged = new ArrayList<>();
		merge(intervals, merged);
		long ans = 0;
		for(int i = 0 ; i < merged.size() ; i++) {
			ans += sum(merged.get(i).len());
		}
		pw.println(ans);
	}
	
	static void merge(ArrayList<pair>a1, ArrayList<pair>a2) {
		int n = a1.size();
		int index = 0;
		for(int i =1 ; i < n ; i++) {
			if(a1.get(index).s >= a1.get(i).f) {
				a1.get(index).s = max(a1.get(index).s, a1.get(i).s);
			}
			else {
				index++;
				a1.set(index, a1.get(i));
			}
		}
		for(int i = 0 ; i <= index ; i++) {
			a2.add(a1.get(i));
		}
//		pw.println(a1);
	}
	
//	int index = 0; // Stores index of last element 
//    // in output array (modified arr[]) 
//
//    // Traverse all input Intervals 
//    for (int i=1; i<arr.length; i++) 
//    { 
//        // If this is not first Interval and overlaps 
//        // with the previous one 
//        if (arr[index].end >=  arr[i].start) 
//        { 
//               // Merge previous and current Intervals 
//            arr[index].end = Math.max(arr[index].end, arr[i].end); 
//        } 
//        else {
//            index++;
//            arr[index] = arr[i]; 
//        }    
//    }
	
	
	
	static pair merge(pair p1, pair p2) {
		return new pair(min(p1.f, p2.f), max(p1.s, p2.s));
	}

	static long sum(long n) {
		return (n * (n + 1) / 2);
	}

	static class pair implements Comparable<pair>{
		int f, s;

		pair(int f, int s) {
			this.f = f;
			this.s = s;
		}

		
		
		
		
		public int len() {
			return s - f + 1;
		}
	}

	public static void main(String[] args) throws IOException {
		Instant start = Instant.now();
		if (args.length == 2) {
			System.setIn(new FileInputStream(new File("D:\\program\\javaCPEclipse\\CodeForces\\src\\input.txt")));
			// System.setOut(new PrintStream(new File("output.txt")));
			System.setErr(new PrintStream(new File("D:\\program\\javaCPEclipse\\CodeForces\\src\\error.txt")));
			DEBUG = true;
		}
		fs = new Reader();
		pw = new PrintWriter(System.out);
		int t = fs.nextInt();
		while (t-- > 0) {
			solve();
		}
		Instant end = Instant.now();
		if (DEBUG) {
			pw.println(Duration.between(start, end));
		}
		pw.close();
	}

	static void sort(int a[]) {
		ArrayList<Integer> l = new ArrayList<Integer>();
		for (int x : a)
			l.add(x);
		Collections.sort(l);
		for (int i = 0; i < a.length; i++) {
			a[i] = l.get(i);
		}
	}

	

	
}
