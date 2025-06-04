import java.io.*;
import java.util.*;

public class D {
	
	static class Pair implements Comparable<Pair>{
		int a;
		char b;

		public Pair(int a, char b) {
			this.a = a;
			this.b = b;
		}

		
	}
	
	public static void main(String[] args)throws IOException {
		
		FastScanner scan = new FastScanner();
		PrintWriter output = new PrintWriter(System.out);
		int t = scan.nextInt();
		for(int tt = 0;tt<t;tt++) {
			
			int n = scan.nextInt();
			int arr[] = scan.readArray(n);
			char line[] = scan.next().toCharArray();
			ArrayList<Pair> pairs = new ArrayList<Pair>();
			for(int i = 0;i<n;i++) {
				pairs.add(new Pair(arr[i], line[i]));
			}
			Collections.sort(pairs);
			boolean possible = true;
			for(int i = 1;i<=n;i++) {
				if(pairs.get(i-1).a == i) {
					continue;
				}
				else if(pairs.get(i-1).a < i && pairs.get(i-1).b == 'R') {
					continue;
				}
				else if(pairs.get(i-1).a > i && pairs.get(i-1).b == 'B') {
					continue;
				}
				else {
					possible = false;
					break;
				}
			}
			output.println(possible == true ? "YES" : "NO");
		}
		output.flush();

	}
	
	public static int[] sort(int arr[]) {
		List<Integer> list = new ArrayList<>();
		for(int i:arr) list.add(i);
		Collections.sort(list);
		for(int i = 0;i<list.size();i++) arr[i] = list.get(i);
		return arr;
	}
	
	
	
	
	
	
	
	

}
