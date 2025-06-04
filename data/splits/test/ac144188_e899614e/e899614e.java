import java.util.*;
import java.io.*;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import static java.lang.Math.max;
public class EdB {
	static long[] mods = {1000000007, 998244353, 1000000009};
	static long mod = mods[0];
	public static MyScanner sc;
    public static PrintWriter out;
	public static void main(String[] largewang) throws Exception{
		// TODO Auto-generated method stub
 		sc = new MyScanner();
 		out = new PrintWriter(System.out);
 		int t = sc.nextInt();
 		while (t-->0) {
 			int n = sc.nextInt();
 			Set<Long> sums = new HashSet<>();
 			boolean found = false;
 			int[] arr = readArrayInt(n);
 			for(int j = 0;j<n;j++){
 				if (arr[j] == 0)
 					found = true;
 			}
 			for(int j = 1;j<(1<<n);j++){
 				long sum = 0;
 				for(int k = 0;k<n;k++){
 					if (((1<<k)&j) != 0) {
 						sum += (long)abs(arr[k]);
 					}
 				}
 				if (sums.contains(sum)) {
 					found = true;
 					break;
 				} else {
 					sums.add(sum);
 				}
 			}
 			verdict(found);
 		}
 		out.close();
 		
 	}
	
	
	
	static int[] readArrayInt(int n){
    	int[] array = new int[n];
    	for(int j = 0;j<n;j++)
    		array[j] = sc.nextInt();
    	return array;
    }
	
	
	
	
	
	
	
	static void verdict(boolean a){
        out.println(a ? "YES" : "NO");
    }
    	
}
