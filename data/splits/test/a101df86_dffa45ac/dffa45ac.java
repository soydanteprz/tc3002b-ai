import java.util.*;
public class Main {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);

		// long mod = 1_000_000_007L;
		// long mod = 998_244_353L;

		int t = sc.nextInt();

		for ( int zzz=0; zzz<t; zzz++ ) {

			int n = sc.nextInt();

			int[] a = new int[n];
			String[] b = new String[n];
			for ( int i=0; i<n; i++ ) {
				a[i] = sc.nextInt();
			}

			int pos = 0;
			if ( n%2==1 ) {
				int a01 = a[0]+a[1];
				if ( a01==0 ) {
					a01 = a[0]-a[1];
					b[0] = String.valueOf(a[2]);
					b[1] = String.valueOf(0-a[2]);
					b[2] = String.valueOf(0-a01);
				} else {
					b[0] = String.valueOf(a[2]);
					b[1] = String.valueOf(a[2]);
					b[2] = String.valueOf(0-a01);
				}

				pos = 3;
			}

			for ( int i=pos; i<n; i=i+2 ) {
				b[i] = String.valueOf(a[i+1]);
				b[i+1] = String.valueOf(0-a[i]);
			}

			System.out.println(String.join(" ", b));
		}
	}
}
