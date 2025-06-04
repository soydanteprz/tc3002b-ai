import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

public class Main {

	public Main() throws FileNotFoundException {

//		File file = Paths.get("input.txt").toFile();
//		if (file.exists()) {
//			System.setIn(new FileInputStream(file));
//		}
		long t = System.currentTimeMillis();

		InputReader reader = new InputReader();

		int ttt = reader.nextInt();

		for (int tt = 0; tt < ttt; tt++) {
			int n=reader.nextInt();
			
			long[] s=new long[n];
			for(int i=0;i<n;i++) {
				s[i]=reader.nextLong();
			}
			
			long smallest1=s[0];
			long smallest2=s[1];
			long val=n*s[0]+n*s[1];
			
			int left1=n-1;
			int left2=n-1;
			long base=s[0]+s[1];
			for(int i=2;i<n;i++) {
				if(i%2==0) {
					//left1
					val=Math.min(val, base+left2*smallest2+left1*s[i]);
					base+=s[i];
					smallest1=Math.min(smallest1, s[i]);
					left1--;
				}else {
					val=Math.min(val, base+left1*smallest1+left2*s[i]);
					base+=s[i];
					smallest2=Math.min(smallest2, s[i]);
					left2--;
					//left2
				}
			}
			

			System.out.println(val);
		}

	}

	

	public static void main(String[] args) throws FileNotFoundException {
		new Main();
	}

}
