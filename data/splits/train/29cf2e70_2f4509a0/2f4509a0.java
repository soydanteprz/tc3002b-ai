import java.util.*;
import java.io.*;
	
public class Main {
		private static FS sc = new FS();
		
		
		
		private static class extra {
			
			static int[] intArr(int size) {
				int[] a = new int[size];
				for(int i = 0; i < size; i++) a[i] = sc.nextInt();
				return a;
			}
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
		}
		
		static int mod = (int) 1e9 + 7;
		
		static LinkedList<String>[] temp, temp2;
		static int inf = (int) 1e9;
//		static long inf = Long.MAX_VALUE;
		
		public static void main(String[] args) {	
			int t = sc.nextInt();
//			int t = 1;
			StringBuilder ret = new StringBuilder();
			while(t-- > 0) {
				int n = sc.nextInt();
				int[] a = extra.intArr(n);
				String s = sc.next();
				ArrayList<Integer> red = new ArrayList<>(), blue = new ArrayList<>();
				for(int i = 0; i < n; i++) {
					if(s.charAt(i) == 'R') red.add(a[i]);
					else blue.add(a[i]);
				}
				Collections.sort(red);
				Collections.sort(blue);
				int start = 1, flag = 0;
				for(int aa:blue) {
					if(start > aa) {
						flag = 1;
						break;
					}
					start++;
				}
				for(int aa:red) {
					if(start < aa) {
						flag = 1;
						break;
					}
					start++;
				}
				if(flag == 1) ret.append("NO\n");
				else ret.append("YES\n");
			}
			System.out.println(ret);
		}
		
	}
