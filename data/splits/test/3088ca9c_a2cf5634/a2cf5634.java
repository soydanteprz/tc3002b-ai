import java.util.*;

import javax.sql.rowset.serial.SerialArray;
import javax.swing.text.html.HTMLDocument.HTMLReader.PreAction;
import java.io.*;
import java.math.*;
import java.sql.Array;;

public class Main {
	

//*************************  FR CLASS ENDS **********************************

	static long mod = (long) (1e9 + 7);

	

	

	

	

	

	

	

//	----------------------------------DSU--------------------------------

	
	static int parent[];
	static int rank[];
	
	
	
	

	/* ***************************************************************************************************************************************************/


	
	
	
	static FR sc = new FR();
	static StringBuilder sb = new StringBuilder();
	
	
	public static void main(String args[]) throws IOException {
		
		int tc = sc.nextInt();
//		int tc = 1;
		while (tc-- > 0) {
			TEST_CASE();
		}
		sb.setLength(sb.length() - 1);
		System.out.println(sb);
	}

	static void TEST_CASE() throws IOException {
		// BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		int n = sc.nextInt();
		String[] strs = new String[n];
		
		for(int i = 0; i < n; i++) {
			strs[i] = sc.next();
		}
		
		int max = -1;
		
		for(int i = 0; i < 5; i++) {
			ArrayList<Integer> arr = new ArrayList<>();
			for(int j = 0; j < n; j++) {
				String s = strs[j];
				char letter = (char)(97+i);
				int val = 0;
				for(int k = 0; k < s.length(); k++) {
					if(s.charAt(k) == letter)
						val++;
					else
						val--;
				}
				arr.add(val);
			}
			Collections.sort(arr, Collections.reverseOrder());
			int val = 0;
			int ans = 0;
			for(int x : arr) {
				val += x;
				if(val <= 0)
					break;
				else
					ans++;
			}
			max = Math.max(ans, max);
		}
		sb.append(max + "\n");
	}
}
