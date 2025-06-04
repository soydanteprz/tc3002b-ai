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
 			char[] a = sc.next().toCharArray();
 			char[] b = sc.next().toCharArray();
 			int counta = 0;
 			int countb = 0;
 			for (char j : a) {
 				if (j == '1')
 					counta++;
 			}
 			for (char j : b) {
 				if (j == '1')
 					countb++;
 			}
 			long ans = 0;
 			if (counta == countb && n-counta+1 == countb){
 				for (int j = 0;j<n;j++){
 					if (b[j] == '1' && a[j] == '0') {
 						ans += 2;
 					}
 				}
 				long tempans = 0;
 				int j=0;
 				for(j = 0;j<n;j++){
 					if (a[j] == '1' && b[j] == '1')
 						break;
 				}
 				for(int i = 0;i<n;i++){
 					if (i == j)
 						continue;
 					if (a[i] == '1')
 						a[i] = '0';
 					else
 						a[i] = '1';
 				}
 				tempans += 1;
 				for (j = 0;j<n;j++){
 					if (b[j] == '1' && a[j] == '0') {
 						tempans += 2;
 					}
 				}
 				ans = min(ans, tempans);
 			} else if (counta == countb) {
 				for (int j = 0;j<n;j++){
 					if (b[j] == '1' && a[j] == '0') {
 						ans += 2;
 					}
 				}
 			} else if (n-counta+1 == countb) {
 				int j=0;
 				for(j = 0;j<n;j++){
 					if (a[j] == '1' && b[j] == '1')
 						break;
 				}
 				for(int i = 0;i<n;i++){
 					if (i == j)
 						continue;
 					if (a[i] == '1')
 						a[i] = '0';
 					else
 						a[i] = '1';
 				}
 				ans += 1;
 				for (j = 0;j<n;j++){
 					if (b[j] == '1' && a[j] == '0') {
 						ans += 2;
 					}
 				}
 			} else {
 				ans = -1;
 			}
 			out.println(ans);
  		}
 		out.close();
 		
 	}
	
	
	
	
	
	
	
	
	
	
	
	
    	
}
